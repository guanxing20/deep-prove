use ff_ext::ExtensionField;
use multilinear_extensions::virtual_poly::VirtualPolynomial;
use sumcheck::structs::IOPProof;

use crate::{
    matrix::Matrix,
    model::{InferenceStep, InferenceTrace, Layer},
};

struct Prover<'a, E: ExtensionField> {
    trace: InferenceTrace<'a, E>,
    proofs: Vec<IOPProof<E>>,
}

impl<'a, E> Prover<'a, E>
where
    E: ExtensionField,
{
    pub fn new(trace: InferenceTrace<'a, E>) -> Self {
        Self {
            trace,
            proofs: Default::default(),
        }
    }
    fn prove_step(&mut self, input: &[E], step: InferenceStep<'a, E>) {
        match step.layer {
            Layer::Dense(matrix) => self.prove_dense_step(input, &step.output, matrix),
        }
    }
    fn prove_dense_step(&mut self, input: &[E], output: &[E], matrix: &Matrix<E>) {
        // contruct the MLE combining the input and the matrix
    }
}
