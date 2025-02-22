use crate::{
    Claim, VectorTranscript,
    commit::{self, precommit, same_poly},
    iop::{StepProof, context::StepInfo, precommit::PolyID},
    lookup::{self, LookupProtocol},
    vector_to_mle,
};
use anyhow::{Context as CC, bail, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
};
use serde::{Serialize, de::DeserializeOwned};
use sumcheck::structs::IOPVerifierState;
use transcript::Transcript;

use super::{
    ActivationProof, Context, DenseProof, Proof,
    context::{ActivationInfo, DenseInfo},
};

/// What the verifier must have besides the proof
pub struct IO<E> {
    /// Input of the inference given to the model
    input: Vec<E>,
    /// Output of the inference
    output: Vec<E>,
}

impl<E> IO<E> {
    pub fn new(input: Vec<E>, output: Vec<E>) -> Self {
        Self { input, output }
    }
}

/// Verifies an inference proof given a context, a proof and the input / output of the model.
pub fn verify<E: ExtensionField, T: Transcript<E>, L: LookupProtocol<E>>(
    ctx: Context<E>,
    proof: Proof<E>,
    io: IO<E>,
    transcript: &mut T,
) -> anyhow::Result<()>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    let mut commit_verifier = precommit::CommitVerifier::new();
    let mut witness_verifier = precommit::CommitVerifier::new();
    ctx.write_to_transcript(transcript)?;
    // 0. Derive the first randomness
    let first_randomness = transcript.read_challenges(io.output.len().ilog2() as usize);
    // 1. For the output, we manually evaluate the MLE and check if it's the same as what prover
    //    gave. Note prover could ellude that but it's simpler to avoid that special check right
    //    now.
    let output_mle = io.output.into_mle();
    let computed_sum = output_mle.evaluate(&first_randomness);
    let mut output_claim = Claim {
        point: first_randomness,
        eval: computed_sum,
    };
    println!(
        "VERIFIER: Proof Order: {:?}",
        proof.steps.iter().map(|p| p.variant_name()).collect_vec()
    );
    // NOTE: if we only had m2v then we need to do the following check manually to make sure the output is correct.
    // For other cases, for example if we have RELU at last, then we _always_ accumulate output claims into the
    // _witness_prover_ part,  so that claim will be verified nonetheless.
    // TODO: optimization to avoid proving the accumulation if last layer is RELU since verifier can do it himself.
    match proof.steps.first().expect("At least one proof") {
        StepProof::Dense(dproof) => {
            // checks that the last g(0) + g(1) is really equal to the output that the verifier's
            // expecting (random evaluation of the output)
            let claimed_sum = dproof.sumcheck.extract_sum();
            ensure!(
                computed_sum == claimed_sum,
                "output vector evaluation is incorrect"
            );
        }
        _ => {}
    }

    // 2. Verify each proof sequentially, Always make sure the proof corresponds to the expected type of proof in the context.
    for (i, (proof, step_kind)) in proof.steps.iter().zip(ctx.steps_info.iter()).enumerate() {
        output_claim = match (proof, step_kind) {
            (StepProof::Activation(proof), StepInfo::Activation(info)) => {
                verify_activation::<_, _, L>(
                    output_claim,
                    proof,
                    info,
                    &mut witness_verifier,
                    transcript,
                )?
            }
            (StepProof::Dense(proof), StepInfo::Dense(info)) => {
                verify_dense(output_claim, proof, info, &mut commit_verifier, transcript)?
            }
            _ => bail!(
                "proof type {} at step {} don't match expected kind {} from setup ",
                proof.variant_name(),
                i,
                step_kind.variant_name()
            ),
        }
    }
    // 3. input verification: evaluating the input at the random evaluation point from the sumcheck
    let input_mle = vector_to_mle(io.input);
    let computed_randomized_input = input_mle.evaluate(&output_claim.point);
    let given_randomized_input = output_claim.eval;
    ensure!(
        computed_randomized_input == given_randomized_input,
        "input not valid from proof"
    );
    // 4. verify the opening of the accumulation of claims
    commit_verifier.verify(&ctx.weights, proof.commit, transcript)?;
    Ok(())
}

fn verify_activation<E: ExtensionField, T: Transcript<E>, L: LookupProtocol<E>>(
    last_claim: Claim<E>,
    proof: &ActivationProof<E>,
    info: &ActivationInfo,
    witness_verifier: &mut commit::precommit::CommitVerifier<E>,
    t: &mut T,
) -> anyhow::Result<Claim<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    // 1. Verify the lookup proof
    let lookup_context = lookup::Context::<E>::new(
        info.multiplicity_num_vars,
        info.num_vars,
        proof.lookup.input_column_claims().len(),
        proof.lookup.output_column_claims().len(),
    );
    let verifier_claims = L::verify(lookup_context, proof.lookup.clone(), t)?;
    // 1. Verify the accumulation proof from last_claim + lookup claim into the new claim
    let sp_ctx = same_poly::Context::<E>::new(info.num_vars);
    let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);
    sp_verifier.add_claim(last_claim)?;
    verifier_claims
        .output_claims()
        .iter()
        .try_for_each(|claim| sp_verifier.add_claim(claim.clone()))?;

    let new_output_claim = sp_verifier.verify(&proof.io_accumulation, t)?;
    // 2. Accumulate the new claim into the witness commitment protocol
    witness_verifier.add_claim(info.poly_id, new_output_claim)?;
    witness_verifier.add_claim(
        info.multiplicity_poly_id,
        verifier_claims.multiplicity_poly_claim().clone(),
    )?;
    // 3. return the input claim for to be proven at subsequent step
    Ok(verifier_claims.input_claims()[0].clone())
}

fn verify_dense<E: ExtensionField, T: Transcript<E>>(
    last_claim: Claim<E>,
    proof: &DenseProof<E>,
    info: &DenseInfo<E>,
    commit_verifier: &mut commit::precommit::CommitVerifier<E>,
    t: &mut T,
) -> anyhow::Result<Claim<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{

    println!("VERIFIER: claim {:?}",last_claim);
    // TODO: currently that API can panic - should remove panic for error
    let subclaim =
        IOPVerifierState::<E>::verify(last_claim.eval, &proof.sumcheck, &info.poly_aux, t);

    // MATRIX OPENING PART
    // pcs_eval means this evaluation should come from a PCS opening proof
    let pcs_eval_input = subclaim
        .point_flat()
        .iter()
        .chain(last_claim.point.iter())
        .cloned()
        .collect_vec();
    // 0 because Matrix comes first in Matrix x Vector
    // Note we don't care about verifying that for the vector since it's verified at the next
    // step.
    let pcs_eval_output = proof.individual_claims[0];
    commit_verifier.add_claim(info.poly_id, Claim::new(pcs_eval_input, pcs_eval_output))?;

    // SUMCHECK verification part
    // Instead of computing the polynomial at the random point requested like this
    // let computed_point = vp.evaluate(
    //     subclaim
    //         .point
    //         .iter()
    //         .map(|c| c.elements)
    //         .collect_vec()
    //         .as_ref(),
    //
    // We compute the evaluation directly from the individual final evaluations of each polynomial
    // involved in the sumcheck the prover's giving,e.g. y(res) = SUM f_i(res)
    ensure!(
        proof.individual_to_virtual_claim() == subclaim.expected_evaluation,
        "sumcheck claim failed",
    );

    // the output claim for this step that is going to be verified at next step
    Ok(Claim {
        // the new randomness to fix at next layer is the randomness from the sumcheck !
        point: subclaim.point_flat(),
        // the claimed sum for the next sumcheck is MLE of the current vector evaluated at the
        // random point. 1 because vector is secondary.
        eval: proof.individual_claims[1],
    })
}
