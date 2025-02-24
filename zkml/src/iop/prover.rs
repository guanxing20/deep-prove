use super::{
    Context, Proof, StepProof,
    context::{ActivationInfo, DenseInfo, StepInfo},
};
use crate::{
    Claim, Element, VectorTranscript,
    activation::{Activation, Relu},
    commit::{precommit, same_poly},
    iop::{ActivationProof, DenseProof},
    logup::{compute_multiplicity_poly, merge_columns},
    lookup::{self, LookupProtocol},
    matrix::Matrix,
    model::{InferenceStep, InferenceTrace, Layer},
    quantization::Requant,
};
use anyhow::{Context as CC, anyhow, bail};
use ff_ext::ExtensionField;
use itertools::Itertools;
use log::{debug, warn};
use mpcs::util::field_type_to_ext_vec;
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    virtual_poly::VirtualPolynomial,
};
use serde::{Serialize, de::DeserializeOwned};
use std::marker::PhantomData;
use sumcheck::structs::{IOPProverState, IOPVerifierState};
use transcript::Transcript;

/// Prover generates a series of sumcheck proofs to prove the inference of a model
pub struct Prover<'a, E: ExtensionField, T: Transcript<E>, L: LookupProtocol<E>>
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
    _phantom: PhantomData<L>,
}

impl<'a, E, T, L> Prover<'a, E, T, L>
where
    T: Transcript<E>,
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    L: LookupProtocol<E>,
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
            _phantom: PhantomData,
        }
    }
    fn prove_step<'b>(
        &mut self,
        last_claim: Claim<E>,
        input: &[E],
        step: &InferenceStep<'b, E>,
        info: &StepInfo<E>,
    ) -> anyhow::Result<Claim<E>> {
        println!("PROVER: proving layer {}", step.layer.describe());
        match (step.layer, info) {
            (Layer::Dense(matrix), StepInfo::Dense(info)) => {
                // NOTE: here we treat the ID of the step AS the ID of the polynomial. THat's okay because we only care
                // about these IDs being unique, so as long as the mapping between poly <-> id is correct, all good.
                // This is the case here since we treat each matrix as a different poly
                self.prove_dense_step(last_claim, input, &step.output, info, matrix)
            }
            (Layer::Activation(Activation::Relu(relu)), StepInfo::Activation(info)) => {
                self.prove_relu(last_claim, input, &step.output, info)
            }
            (Layer::Requant(info), StepInfo::Requant(info2)) => {
                self.prove_requant(last_claim, input, &step.output, info)
            }
            _ => bail!(
                "inconsistent proof step {} and info step {} from ctx",
                step.layer.describe(),
                info.variant_name()
            ),
        }
    }

    fn prove_requant(
        &self,
        last_claim: Claim<E>,
        input: &[E],
        output: &[E],
        info: &Requant,
    ) -> anyhow::Result<Claim<E>> {
        unimplemented!()
    }

    fn prove_relu(
        &mut self,
        last_claim: Claim<E>,
        // input to the relu
        input: &[E],
        // output of the relu
        output: &[E],
        info: &ActivationInfo,
    ) -> anyhow::Result<Claim<E>> {
        assert_eq!(
            input.len(),
            output.len(),
            "input/output of lookup don't have same size"
        );

        // Debug check to see that all the pairs looked up are rows in the table
        debug_assert!({
            let (relu_one, relu_two) = Relu::to_mle::<E>();

            let mut bool_out = true;
            for (in_val, out_val) in input.iter().zip(output.iter()) {
                let pos = relu_one
                    .iter()
                    .position(|relu_one_val| *relu_one_val == *in_val)
                    .ok_or(anyhow!("Input value was not in the table: {:?}", in_val))?;
                debug_assert_eq!(relu_two[pos], *out_val);

                bool_out ^= relu_two[pos] == *out_val;
            }
            bool_out
        });

        // First call the lookup with the right arguments:
        // * table mle: one mle per column
        // * lookup mle: one mle per column, where the evals are just the list of inputs and output ordered by access
        let lookup_num_vars = input.len().ilog2() as usize;
        let table_mles = self
            .ctx
            .activation
            .relu_polys()
            .iter()
            .map(|evals| {
                DenseMultilinearExtension::from_evaluations_vec(
                    Relu::num_vars(),
                    evals
                        .iter()
                        .map(|val| val.as_bases()[0])
                        .collect::<Vec<E::BaseField>>(),
                )
            })
            .collect::<Vec<DenseMultilinearExtension<E>>>();
        let lookup_mles = vec![
            DenseMultilinearExtension::<E>::from_evaluations_vec(
                lookup_num_vars,
                input
                    .iter()
                    .map(|val| val.as_bases()[0])
                    .collect::<Vec<E::BaseField>>(),
            ),
            DenseMultilinearExtension::<E>::from_evaluations_vec(
                lookup_num_vars,
                output
                    .iter()
                    .map(|val| val.as_bases()[0])
                    .collect::<Vec<E::BaseField>>(),
            ),
        ];

        // TODO: replace via proper lookup protocol
        let lookup_ctx =
            lookup::Context::<E>::new(Relu::num_vars(), input.len().ilog2() as usize, 1, 1);
        let lookup_proof = L::prove(&lookup_ctx, table_mles, lookup_mles, self.transcript)?;
        // in our case, the output of the RELU is ALSO the same poly that previous proving
        // step (likely dense) has "outputted" to evaluate at a random point. So here we accumulate the two claims,
        // the one from previous proving step and the one given by the lookup protocol into one. Since they're claims
        // about the same poly, we can use the "same_poly" protocol.
        let same_poly_ctx = same_poly::Context::<E>::new(info.num_vars);
        let mut same_poly_prover = same_poly::Prover::<E>::new(output.to_vec().into_mle());
        same_poly_prover.add_claim(last_claim)?;
        let input_claim = lookup_proof.input_column_claims()[0].clone();
        let output_claim = lookup_proof.output_column_claims()[0].clone();

        same_poly_prover.add_claim(output_claim)?;
        let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, self.transcript)?;
        // order is (output,mult)
        self.witness_prover
            .add_claim(info.poly_id, claim_acc_proof.extract_claim())?;
        self.witness_prover.add_claim(
            info.multiplicity_poly_id,
            lookup_proof.multiplicity_claim().clone(),
        )?;
        // the next step is gonna take care of proving the next claim
        // TODO: clarify that bit - here cheating because of fake lookup protocol, but inconsistency
        // between padded input and real inputsize. Next proving step REQUIRES the real input size.
        let next_claim = {
            let input_mle = input.to_vec().into_mle();
            let point = input_claim.pad(input_mle.num_vars()).point;
            let eval = input_mle.evaluate(&point);
            Claim::new(point, eval)
        };

        // Add the proof in
        self.proofs.push(StepProof::Activation(ActivationProof {
            io_accumulation: claim_acc_proof,
            lookup: lookup_proof,
        }));
        Ok(next_claim)
    }

    fn prove_dense_step(
        &mut self,
        // last random claim made
        last_claim: Claim<E>,
        // input to the dense layer
        input: &[E],
        // output of dense layer evaluation
        output: &[E],
        info: &DenseInfo<E>,
        matrix: &Matrix<Element>,
    ) -> anyhow::Result<Claim<E>> {
        println!("PROVER: claim {:?}", last_claim);
        let (nrows, ncols) = (matrix.nrows(), matrix.ncols());
        assert_eq!(
            nrows,
            output.len(),
            "dense proving: nrows {} vs output {}",
            nrows,
            output.len()
        );
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
        let input_mle = input.to_vec().into_mle();

        assert_eq!(mat_mle.num_vars(), input_mle.num_vars());
        let num_vars = input_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
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
            let mut vp = VirtualPolynomial::<E>::new(num_vars);
            vp.add_mle_list(vec![mat_mle.into(), input_mle.into()], E::ONE);
            // asserted_sum in this case is the output MLE evaluated at the random point
            let mle_output = output.to_vec().into_mle();
            let claimed_sum = mle_output.evaluate(&last_claim.point);
            debug_assert_eq!(claimed_sum, last_claim.eval, "sumcheck eval weird");
            debug_assert_eq!(claimed_sum, proof.extract_sum(), "sumcheck output weird");

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
            .add_claim(info.poly_id, Claim::new(point, eval))
            .context("unable to add claim")?;

        // the claim that this proving step outputs is the claim about not the matrix but the vector poly.
        // at next step, that claim will be proven over this vector poly (either by the next dense layer proving, or RELU etc).
        let claim = Claim {
            point: proof.point.clone(),
            eval: state.get_mle_final_evaluations()[1],
        };
        self.proofs.push(StepProof::Dense(DenseProof {
            sumcheck: proof,
            individual_claims: state.get_mle_final_evaluations(),
        }));
        Ok(claim)
    }

    pub fn prove<'b>(mut self, trace: InferenceTrace<'b, E>) -> anyhow::Result<Proof<E>> {
        // First, create the context for the witness polys -
        self.instantiate_witness_ctx(&trace, &self.ctx.steps_info)?;
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
        let y_i = trace.last_step().output.clone().into_mle().evaluate(&r_i);
        let mut last_claim = Claim {
            point: r_i,
            eval: y_i,
        };
        // we start by the output to prove up to the input, GKR style
        for (i, ((input, step), info)) in trace
            .iter()
            .rev()
            .zip(self.ctx.steps_info.iter())
            .enumerate()
        {
            last_claim = self.prove_step(last_claim, input, step, &info)?;
        }
        // now provide opening proofs for all claims accumulated during the proving steps
        let commit_proof = self
            .commit_prover
            .prove(&self.ctx.weights, self.transcript)?;
        let mut output_proof = Proof {
            steps: self.proofs,
            commit: commit_proof,
            witness: None,
        };
        if let Some(witness_ctx) = self.witness_ctx {
            let witness_proof = self.witness_prover.prove(&witness_ctx, self.transcript)?;
            output_proof.witness = Some((witness_proof, witness_ctx));
        }
        Ok(output_proof)
    }

    /// Looks at all the individual polys to accumulate from the witnesses and create the context from that.
    fn instantiate_witness_ctx<'b>(
        &mut self,
        trace: &InferenceTrace<'b, E>,
        step_infos: &[StepInfo<E>],
    ) -> anyhow::Result<()> {
        let polys = trace
            .iter()
            .rev()
            .zip(step_infos.iter())
            .filter_map(|((input, step), info)| {
                match (step.layer, info) {
                    (Layer::Activation(Activation::Relu(_)), StepInfo::Activation(info)) => {
                        let (table_in, table_out) = Relu::to_mle::<E>();
                        let table_columns = [table_in, table_out].map(|evals| {
                            DenseMultilinearExtension::from_evaluations_vec(
                                Relu::num_vars(),
                                evals
                                    .iter()
                                    .map(|val| val.as_bases()[0])
                                    .collect::<Vec<E::BaseField>>(),
                            )
                        });
                        let lookup_columns = [input, step.output.as_slice()].map(|evals| {
                            DenseMultilinearExtension::<E>::from_evaluations_vec(
                                info.num_vars,
                                evals
                                    .iter()
                                    .map(|val| val.as_bases()[0])
                                    .collect::<Vec<E::BaseField>>(),
                            )
                        });

                        let merged_table = merge_columns(&table_columns, E::ONE);
                        let merged_lookup = merge_columns(&lookup_columns, E::ONE);

                        let m_poly = compute_multiplicity_poly::<E>(&merged_table, &merged_lookup);
                        let multiplicity_poly_evals = field_type_to_ext_vec(m_poly.evaluations());

                        Some(vec![
                            (info.poly_id, step.output.clone()),
                            (info.multiplicity_poly_id, multiplicity_poly_evals),
                        ])
                    }
                    // the dense layer is handling everything "on its own"
                    _ => None,
                }
            })
            .flatten()
            .collect_vec();
        if !polys.is_empty() {
            let ctx = precommit::Context::generate(polys)
                .context("unable to generate ctx for witnesses")?;
            self.witness_ctx = Some(ctx);
        } else {
            warn!("no activation functions found - no witness commitment");
        }
        Ok(())
    }
}

/// Pad all inner vectors to the given size
fn pad2<E: Clone>(a: Vec<Vec<E>>, nsize: usize, with: E) -> Vec<Vec<E>> {
    // check vectors inside are all of the same length respectively
    assert_eq!(
        a.iter().map(|v| v.len()).sum::<usize>(),
        a.len() * a[0].len()
    );
    // make sure we're not doing anything wrong
    assert!(a.iter().all(|v| v.len() <= nsize));
    a.into_iter()
        .map(|mut v| {
            v.resize(nsize, with.clone());
            v
        })
        .collect_vec()
}

#[cfg(test)]
mod test {

    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};

    use crate::{
        Claim,
        testing::{random_field_vector, random_vector},
    };

    use ff::Field;
    type F = GoldilocksExt2;

    #[test]
    fn test_padding_prover() {
        let num_vars = 7;
        let poly_size = 1 << num_vars;
        let padded_num_vars = 10;
        let padded_size = 1 << padded_num_vars;
        let poly = random_field_vector(1 << num_vars);
        let padded_poly = poly
            .iter()
            .chain(std::iter::repeat(&F::ZERO))
            .take(padded_size)
            .cloned()
            .collect_vec();
        let padded_point = random_field_vector::<F>(padded_num_vars);
        let padded_eval = padded_poly.into_mle().evaluate(&padded_point);
        // now resize the claim to the original poly size (emulating what next dense layer proving is doing)
        let reduced_point = padded_point.iter().take(num_vars).cloned().collect_vec();
        let eval = poly.into_mle().evaluate(&reduced_point);
        assert_eq!(padded_eval, eval);
    }
}
