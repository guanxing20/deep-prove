use std::cmp::max;

use ark_std::rand::random;
use ff_ext::ExtensionField;
use itertools::Itertools;
use log::info;
use multilinear_extensions::{mle::MultilinearExtension, virtual_poly::VirtualPolynomial};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use transcript::{BasicTranscript, Transcript};

use crate::{
    matrix::Matrix,
    model::{InferenceStep, InferenceTrace, Layer},
    vector_to_mle,
};

struct Prover<E: ExtensionField> {
    proofs: Vec<IOPProof<E>>,
    transcript: BasicTranscript<E>,
}

pub fn default_transcript<E: ExtensionField>() -> BasicTranscript<E> {
    BasicTranscript::new(b"m2vec")
}

impl<E> Prover<E>
where
    E: ExtensionField,
{
    pub fn new() -> Self {
        Self {
            proofs: Default::default(),
            transcript: default_transcript(),
        }
    }
    fn prove_step<'a>(
        &mut self,
        random_vars_to_fix: Vec<E>,
        input: &[E],
        step: &InferenceStep<'a, E>,
    ) {
        match step.layer {
            Layer::Dense(matrix) => {
                self.prove_dense_step(random_vars_to_fix, input, &step.output, matrix)
            }
        }
    }
    fn prove_dense_step(
        &mut self,
        random_vars_to_fix: Vec<E>,
        input: &[E],
        output: &[E],
        matrix: &Matrix<E>,
    ) {
        let (nrows, ncols) = (matrix.nrows(), matrix.ncols());
        assert_eq!(nrows, output.len(), "something's wrong with the output");
        assert_eq!(
            nrows.ilog2() as usize,
            random_vars_to_fix.len(),
            "something's wrong with the randomness"
        );
        assert_eq!(ncols, input.len(), "something's wrong with the input");
        // contruct the MLE combining the input and the matrix
        let mut mat_mle = matrix.to_mle();
        // fix the variables from the random input
        // NOTE: here we must fix the HIGH variables because the MLE is addressing in little
        // endian so (rows,cols) is actually given in (cols, rows)
        // mat_mle.fix_variables_in_place_parallel(partial_point);
        println!("mat_mle before fixing: {}", mat_mle.num_vars());
        mat_mle.fix_high_variables_in_place(&random_vars_to_fix);
        println!("mat_mle after fixing: {}", mat_mle.num_vars());
        let input_mle = vector_to_mle(input.to_vec());
        println!("INPUT num vars {}", input_mle.num_vars());
        let max_var = max(mat_mle.num_vars(), input_mle.num_vars());
        let mut vp = VirtualPolynomial::<E>::new(max_var);
        // TODO: remove the clone once prover+verifier are working
        vp.add_mle_list(
            vec![mat_mle.clone().into(), input_mle.clone().into()],
            E::ONE,
        );
        let (proof, _) = IOPProverState::<E>::prove_parallel(vp, &mut self.transcript);

        debug_assert!({
            let mut t = default_transcript::<E>();
            // asserted_sum in this case is the output MLE evaluated at the random point
            let mle_output = vector_to_mle(output.to_vec());
            let claimed_sum = mle_output.evaluate(&random_vars_to_fix);
            // just construct manually here instead of cloning in the non debug code
            let mut vp = VirtualPolynomial::<E>::new(max_var);
            vp.add_mle_list(vec![mat_mle.into(), input_mle.into()], E::ONE);
            let subclaim = IOPVerifierState::<E>::verify(claimed_sum, &proof, &vp.aux_info, &mut t);
            // now assert that the polynomial evaluated at the random point of the sumcheck proof
            // is equal to what the sumcheck proof is saying. This step can be done via PCS opening
            // proofs for all steps but first (output of inference) and last (input of inference)
            let computed_point = vp.evaluate(
                subclaim
                    .point
                    .iter()
                    .map(|c| c.elements)
                    .collect_vec()
                    .as_ref(),
            );

            // NOTE: this expected_evaluation is computed by the verifier on the "reduced"
            // last polynomial of the sumcheck protocol. It's easy to compute since it's a degree
            // one poly. However, it needs to be checked against the original polynomial and this
            // should/usually done via PCS.
            computed_point == subclaim.expected_evaluation
        });

        self.proofs.push(proof);
    }

    pub fn prove<'a>(&mut self, trace: InferenceTrace<'a, E>) {
        // TODO: input the commitments first to do proper FS

        // this is the random set of variables to fix at each step derived as the output of
        // sumcheck.
        // For the first step, so before the first sumcheck, we generate it from FS.
        // The dimension is simply the number of variables needed to address all the space of the
        // input vector.
        let mut randomness_to_fix = (0..trace.final_output().len().ilog2() as usize)
            .map(|_| self.transcript.read_challenge().elements)
            .collect_vec();

        // we start by the output to prove up to the input, GKR style
        for (i, (input, step)) in trace.iter().rev().enumerate() {
            info!(
                "step {}: input.len = {:?}, step.matrix {:?}, step.output.len() = {:?}",
                i,
                input.len(),
                step.layer.dim(),
                step.output.len()
            );
            self.prove_step(randomness_to_fix, input, step);
            // this point is the last random point over which to evaluate the original polynomial.
            // In our case, the polynomial is actually 2 for dense layer: the matrix MLE and the
            // vector MLE.
            //
            // So normally the verifier should verify both of these poly at this point. However:
            // 1. For the matrix MLE, we rely on PCS opening proofs, which is another step of the
            //    prover flow.
            // 2. For the vector, we actually do a subsequent sumcheck to prove that the vector MLE
            //    is really equal to the claimed evaluation.
            randomness_to_fix = self.proofs.last().unwrap().point.clone();
        }
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use crate::model::Model;

    use super::Prover;

    type F = GoldilocksExt2;
    use tracing_subscriber;

    #[test]
    fn test_prover_steps() {
        tracing_subscriber::fmt::init();
        let (model, input) = Model::<F>::random(2);
        let trace = model.run(input);
        let mut prover = Prover::new();
        prover.prove(trace);
    }
}
