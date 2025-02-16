use super::{Context, Proof, StepProof};
use crate::{
    Claim, Element, VectorTranscript,
    activation::Activation,
    iop::{Matrix2VecProof, precommit, precommit::PolyID, same_poly},
    lookup,
    lookup::LookupProtocol,
    matrix::Matrix,
    model::{InferenceStep, InferenceTrace, Layer, StepIdx},
    vector_to_mle,
};
use anyhow::Context as CC;
use ff_ext::ExtensionField;
use itertools::Itertools;
use log::debug;
use multilinear_extensions::{
    mle::{IntoMLE, IntoMLEs, MultilinearExtension},
    virtual_poly::VirtualPolynomial,
};
use serde::{Serialize, de::DeserializeOwned};
use std::cmp::max;
use sumcheck::structs::{IOPProverState, IOPVerifierState};
use transcript::Transcript;

/// Prover generates a series of sumcheck proofs to prove the inference of a model
pub struct Prover<'a, E: ExtensionField, T: Transcript<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    ctx: &'a Context<E>,
    // proofs for each layer being filled
    proofs: Vec<StepProof<E>>,
    transcript: &'a mut T,
    commit_prover: precommit::CommitProver<E>,
    /// the context of the witness part (IO of lookups, linked with matrix2vec for example)
    /// is generated during proving time. It is first generated and then the fiat shamir starts.
    /// The verifier doesn't know about the individual polys (otherwise it beats the purpose) so
    /// that's why it is generated at proof time.
    witness_ctx: Option<precommit::Context<E>>,
    /// The prover related to proving multiple claims about different witness polyy (io of lookups etc)
    witness_prover: precommit::CommitProver<E>,
}

impl<'a, E, T> Prover<'a, E, T>
where
    T: Transcript<E>,
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub fn new(ctx: &'a Context<E>, transcript: &'a mut T) -> Self {
        Self {
            ctx,
            transcript,
            proofs: Default::default(),
            commit_prover: precommit::CommitProver::new(),
            // at this step, we can't build the ctx since we don't know the individual polys
            witness_ctx: None,
            witness_prover: precommit::CommitProver::new(),
        }
    }
    fn prove_step<'b>(
        &mut self,
        last_claim: Claim<E>,
        input: &[E],
        step: &InferenceStep<'b, E>,
    ) -> anyhow::Result<Claim<E>> {
        match step.layer {
            Layer::Dense(matrix) => {
                // NOTE: here we treat the ID of the step AS the ID of the polynomial. THat's okay because we only care
                // about these IDs being unique, so as long as the mapping between poly <-> id is correct, all good.
                // This is the case here since we treat each matrix as a different poly
                self.prove_dense_step(last_claim, input, &step.output, (step.id as PolyID, matrix))
            }
            Layer::Activation(crate::activation::Activation::Relu(relu)) => {
                self.prove_relu(last_claim, input, &step.output, step.id)
            }
        }
    }

    fn prove_relu(
        &mut self,
        last_claim: Claim<E>,
        // input to the relu
        input: &[E],
        // output of the relu
        output: &[E],
        // the step_id is used to associate the polys to accumulate for this lookup argument
        step_id: StepIdx,
    ) -> anyhow::Result<Claim<E>> {
        // First call the lookup with the right arguments:
        // * table mle: one mle per column
        // * lookup mle: one mle per column, where the evals are just the list of inputs and output ordered by access
        let table_mles = self.ctx.activation.relu_polys().into_mles();
        let lookup_mles = vec![input.to_vec(), output.to_vec()].into_mles();
        // TODO: replace via proper lookup protocol
        let mut lookup_proof =
            lookup::DummyLookup::prove(table_mles, lookup_mles, self.transcript)?;
        // in our case, the output of the RELU is ALSO the same poly that previous proving
        // step (likely dense) has "outputted" to evaluate at a random point. So here we accumulate the two claims,
        // the one from previous proving step and the one given by the lookup protocol into one. Since they're claims
        // about the same poly, we can use the "same_poly" protocol.
        let same_poly_ctx = same_poly::Context::<E>::new(output.len());
        let mut same_poly_prover = same_poly::Prover::<E>::new(output.to_vec().into_mle());
        same_poly_prover.add_claim(last_claim)?;
        let (input_claim, output_claim) =
            (lookup_proof.claims.remove(0), lookup_proof.claims.remove(0));
        same_poly_prover.add_claim(output_claim)?;
        let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, self.transcript)?;
        // order is (output,mult)
        // TODO: add multiplicities, etc...
        self.witness_prover
            .add_claim(step_id * 3, claim_acc_proof.extract_claim())?;
        // the next step is gonna take care of proving the next claim
        Ok(input_claim)
    }

    fn prove_dense_step(
        &mut self,
        // last random claim made
        last_claim: Claim<E>,
        // input to the dense layer
        input: &[E],
        // output of dense layer evaluation
        output: &[E],
        (id, matrix): (PolyID, &Matrix<Element>),
    ) -> anyhow::Result<Claim<E>> {
        let (nrows, ncols) = (matrix.nrows(), matrix.ncols());
        assert_eq!(nrows, output.len(), "something's wrong with the output");
        assert_eq!(
            nrows.ilog2() as usize,
            last_claim.point.len(),
            "something's wrong with the randomness"
        );
        assert_eq!(ncols, input.len(), "something's wrong with the input");
        // contruct the MLE combining the input and the matrix
        let mut mat_mle = matrix.to_mle();
        // fix the variables from the random input
        // NOTE: here we must fix the HIGH variables because the MLE is addressing in little
        // endian so (rows,cols) is actually given in (cols, rows)
        // mat_mle.fix_variables_in_place_parallel(partial_point);
        mat_mle.fix_high_variables_in_place(&last_claim.point);
        let input_mle = vector_to_mle(input.to_vec());
        let max_var = max(mat_mle.num_vars(), input_mle.num_vars());
        assert_eq!(mat_mle.num_vars(), input_mle.num_vars());
        let mut vp = VirtualPolynomial::<E>::new(max_var);
        // TODO: remove the clone once prover+verifier are working
        vp.add_mle_list(
            vec![mat_mle.clone().into(), input_mle.clone().into()],
            E::ONE,
        );
        let tmp_transcript = self.transcript.clone();
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, self.transcript);

        debug_assert!({
            let mut t = tmp_transcript;
            // just construct manually here instead of cloning in the non debug code
            let mut vp = VirtualPolynomial::<E>::new(max_var);
            vp.add_mle_list(vec![mat_mle.into(), input_mle.into()], E::ONE);
            // asserted_sum in this case is the output MLE evaluated at the random point
            let mle_output = vector_to_mle(output.to_vec());
            let claimed_sum = mle_output.evaluate(&last_claim.point);
            debug_assert_eq!(claimed_sum, proof.extract_sum(), "sumcheck output weird");
            debug_assert_eq!(claimed_sum, last_claim.eval);

            debug!("prover: claimed sum: {:?}", claimed_sum);
            let subclaim = IOPVerifierState::<E>::verify(claimed_sum, &proof, &vp.aux_info, &mut t);
            // now assert that the polynomial evaluated at the random point of the sumcheck proof
            // is equal to last small poly sent by prover (`subclaim.expected_evaluation`). This
            // step can be done via PCS opening proofs for all steps but first (output of
            // inference) and last (input of inference)
            let computed_point = vp.evaluate(subclaim.point_flat().as_ref());

            let final_prover_point = state
                .get_mle_final_evaluations()
                .into_iter()
                .fold(E::ONE, |acc, eval| acc * eval);
            assert_eq!(computed_point, final_prover_point);

            // NOTE: this expected_evaluation is computed by the verifier on the "reduced"
            // last polynomial of the sumcheck protocol. It's easy to compute since it's a degree
            // one poly. However, it needs to be checked against the original polynomial and this
            // is done via PCS.
            computed_point == subclaim.expected_evaluation
        });

        // PCS part: here we need to create an opening proof for the final evaluation of the matrix poly
        // Note we need the _full_ input to the matrix since the matrix MLE has (row,column) vars space
        let point = [proof.point.as_slice(), last_claim.point.as_slice()].concat();
        let eval = state.get_mle_final_evaluations()[0];
        self.commit_prover
            .add_claim(id, Claim::from(point, eval))
            .context("unable to add claim")?;

        // the claim that this proving step outputs is the claim about not the matrix but the vector poly.
        // at next step, that claim will be proven over this vector poly (either by the next dense layer proving, or RELU etc).
        let claim = Claim {
            point: proof.point.clone(),
            eval: state.get_mle_final_evaluations()[1],
        };
        self.proofs.push(StepProof::M2V(Matrix2VecProof {
            proof: proof,
            individual_claims: state.get_mle_final_evaluations(),
        }));
        Ok(claim)
    }

    pub fn prove<'b>(mut self, trace: InferenceTrace<'b, E>) -> anyhow::Result<Proof<E>> {
        // First, create the context for the witness polys -
        self.instantiate_witness_ctx(&trace)?;
        // write commitments and polynomials info to transcript
        self.ctx.write_to_transcript(self.transcript)?;
        // this is the random set of variables to fix at each step derived as the output of
        // sumcheck.
        // For the first step, so before the first sumcheck, we generate it from FS.
        // The dimension is simply the number of variables needed to address all the space of the
        // input vector.
        let r_i = self
            .transcript
            .read_challenges(trace.final_output().len().ilog2() as usize);
        let y_i = vector_to_mle(trace.last_step().output.clone()).evaluate(&r_i);
        let mut last_claim = Claim {
            point: r_i,
            eval: y_i,
        };
        // we start by the output to prove up to the input, GKR style
        for (i, (input, step)) in trace.iter().rev().enumerate() {
            last_claim = self.prove_step(last_claim, input, step)?;
        }
        // now provide opening proofs for all claims accumulated during the proving steps
        let commit_proof = self
            .commit_prover
            .prove(&self.ctx.weights, self.transcript)?;
        Ok(Proof {
            steps: self.proofs,
            commit: commit_proof,
        })
    }

    /// Looks at all the individual polys to accumulate from the witnesses and create the context from that.
    fn instantiate_witness_ctx<'b>(&mut self, trace: &InferenceTrace<'b, E>) -> anyhow::Result<()> {
        let polys = trace
            .iter()
            .rev()
            .filter_map(|(input, step)| {
                match step.layer {
                    Layer::Activation(Activation::Relu(_)) => {
                        // TODO: right now we accumulate output, also need to accumulate the multiplicities poly
                        //       need to accumulate the other polys from lookup table
                        // We need distinct poly id and at each step we "accumulate" 3 poly
                        // we can "expand" the set of IDs of polys we commit to here since they're proven fully separated
                        // from the matrices weight IDs.
                        let base_id = step.id * 3;
                        // Some(vec![(base_id + 0, input.to_vec()),(base_id + 1, step.output.clone())])
                        Some(vec![(base_id + 1, step.output.clone())])
                    }
                    // the dense layer is handling everything "on its own"
                    Layer::Dense(_) => None,
                }
            })
            .flatten()
            .collect_vec();
        if !polys.is_empty() {
            let ctx = precommit::Context::generate(polys)
                .context("unable to generate ctx for witnesses")?;
            self.witness_ctx = Some(ctx);
        }
        Ok(())
    }
}
