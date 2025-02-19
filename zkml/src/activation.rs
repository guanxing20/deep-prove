use ff_ext::ExtensionField;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::Element;

/// Context holding information related to the lookup tables used in the proving
/// steps for the verifier and the prover.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationCtx<E: ExtensionField> {
    relu_mle_input: Vec<E>,
    relu_mle_output: Vec<E>,
}

impl<E: ExtensionField> ActivationCtx<E> {
    pub fn new() -> Self {
        let (inp, out) = Relu::to_mle();
        Self {
            relu_mle_input: inp,
            relu_mle_output: out,
        }
    }
    /// Returns the input and output MLE of the RELU table
    pub fn relu_polys(&self) -> Vec<Vec<E>> {
        vec![self.relu_mle_input.clone(), self.relu_mle_output.clone()]
    }
}

#[derive(Clone, Debug)]
pub enum Activation {
    Relu(Relu),
}

impl Activation {
    pub fn op(&self, input: &[Element]) -> Vec<Element> {
        match self {
            Activation::Relu(relu) => relu.op(input),
        }
    }
}

/// RELU over 16 bit for 8 bit quantization (after a dense layer, a number could be up to 16bits)
/// ASSUMPTIONS about quantization: [-1;1] is mapped to [0;256]
/// so over 16 bits, middle "0" is 2**15
/// TODO: add quantization
const BIT_QUANTIZATION: usize = 8;
pub const ZERO_QUANTIZED: Element = 1 << 15;
#[derive(Clone, Debug)]
pub struct Relu;

impl Relu {
    pub fn new() -> Relu {
        Self
    }
    pub fn num_vars() -> usize {
        BIT_QUANTIZATION * 2
    }
    pub fn poly_len() -> usize {
        1 << Self::num_vars()
    }
    pub fn shape() -> Vec<usize> {
        vec![2, Self::poly_len()]
    }
    /// to_mle returns two polynomials:
    /// f_i: one containing the input column values
    /// f_o: one containing the output column values
    pub fn to_mle<E: ExtensionField>() -> (Vec<E>, Vec<E>) {
        let num_vars = BIT_QUANTIZATION * 2;
        (0..1 << num_vars)
            .map(|i| (E::from(i), E::from(Self::apply(i))))
            .unzip()
    }

    pub fn op(&self, input: &[Element]) -> Vec<Element> {
        input
            .par_iter()
            .map(|e| Self::apply(*e))
            .collect::<Vec<_>>()
    }

    #[inline(always)]
    pub fn apply(e: Element) -> Element {
        if e <= ZERO_QUANTIZED {
            0
        } else {
            // we also apply requantization by reshifting into [0;2**8] range
            e >> 8
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{testing::random_vector, to_bit_sequence_le};
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};

    use super::*;

    type F = GoldilocksExt2;

    #[test]
    fn test_activation_relu_apply() {
        struct testCase {
            input: Element,
            output: Element,
        }

        impl testCase {
            pub fn from(input: Element, output: Element) -> Self {
                Self { input, output }
            }
        }
        for case in [
            testCase::from(10, 0),
            testCase::from(255, 0),
            testCase::from(4031, 15),
        ] {
            assert_eq!(Relu::apply(case.input), case.output);
        }
    }

    #[test]
    fn test_activation_relu_mle() {
        let ctx = ActivationCtx::<F>::new();
        let relu = Relu::new();
        let mut table_mle = ctx.relu_polys();
        let (input_poly, output_poly) = (table_mle.remove(0), table_mle.remove(0));
        assert_eq!(input_poly.len(), output_poly.len());
        let (input_mle, output_mle) = (
            DenseMultilinearExtension::from_evaluation_vec_smart(
                Relu::num_vars(),
                input_poly.to_vec(),
            ),
            DenseMultilinearExtension::from_evaluation_vec_smart(
                Relu::num_vars(),
                output_poly.to_vec(),
            ),
        );
        assert_eq!(input_mle.num_vars(), output_mle.num_vars());
        assert_eq!(input_mle.num_vars(), Relu::num_vars());
        let inputs = random_vector(10);
        let outputs = relu.op(&inputs);
        assert_eq!(inputs.len(), outputs.len());
        for (idx, (input, output)) in inputs.iter().zip(outputs.iter()).enumerate() {
            // here putting input works because every random input is a u8, so it's already within [0;256] so
            // its value "is" the index. Normally if this is not true, we should get the index of the row corresponding to that input
            let idx_vars = to_bit_sequence_le(*input as usize, Relu::num_vars())
                .map(|b| F::from(b as u64))
                .collect_vec();
            let input_field = input_mle.evaluate(&idx_vars);
            assert_eq!(input_field, F::from(*input));
            let output_field = output_mle.evaluate(&idx_vars);
            assert_eq!(output_field, F::from(*output));
        }
        // assert_eq!(expected,given);
    }
}
