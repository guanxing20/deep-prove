use ff_ext::ExtensionField;
use rayon::iter::ParallelIterator;
use itertools::Itertools;
use multilinear_extensions::mle::DenseMultilinearExtension;
use rayon::iter::IntoParallelRefIterator;

use crate::Element;

#[derive(Clone,Debug)]
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

/// RELU over 16 bit for 8 bit quantization
/// ASSUMPTIONS about quantization: [-1;1] is mapped to [0;256]
/// The "0" is therefore = 127 
/// TODO: add quantization
const BIT_QUANTIZATION: usize = 8;
pub const ZERO_QUANTIZED: Element = 127;
#[derive(Clone,Debug)]
pub struct Relu;


impl Relu {
    pub fn new() -> Relu {
        Self
    }
    pub fn num_vars() -> usize {
        BIT_QUANTIZATION * 2
    }
    /// to_mle returns two polynomials:
    /// f_i: one containing the input column values
    /// f_o: one containing the output column values
    pub fn to_mle<E: ExtensionField>(&self) -> (Vec<E>,Vec<E>) {
        let num_vars = BIT_QUANTIZATION*2;
        (0..1 << num_vars).map(|i| {
            (E::from(i), if i < ZERO_QUANTIZED {
                E::ZERO
            } else {
                E::from(i as u64) 
            })
        }).unzip()

    }
    pub fn op(&self, input: &[Element]) -> Vec<Element> {
        input.par_iter().map(|e|Self::apply(*e)).collect::<Vec<_>>()
    }

    #[inline(always)]
    pub fn apply(e: Element) -> Element {
        if e < ZERO_QUANTIZED {
            0
        } else {
            e
        }
        
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;
    use multilinear_extensions::mle::MultilinearExtension;
    use crate::{testing::random_vector, to_bit_sequence_le, vector_to_field_par};

    use super::*;

    type F = GoldilocksExt2;

    #[test]
    fn test_activation_relu() {
        let relu = Relu::new();
        let (input_poly,output_poly) = relu.to_mle::<F>();
        assert_eq!(input_poly.len(),output_poly.len());
        let (input_mle,output_mle) = (
            DenseMultilinearExtension::from_evaluation_vec_smart(Relu::num_vars(), 
            input_poly),DenseMultilinearExtension::from_evaluation_vec_smart(Relu::num_vars(), output_poly)
        );
        assert_eq!(input_mle.num_vars(),output_mle.num_vars());
        assert_eq!(input_mle.num_vars(),Relu::num_vars());
        let inputs = random_vector(10);
        let outputs = relu.op(&inputs);
        assert_eq!(inputs.len(),outputs.len());
        for (idx, (input,output)) in inputs.iter().zip(outputs.iter()).enumerate() {
            // here putting input works because every random input is a u8, so it's already within [0;256] so 
            // its value "is" the index. Normally if this is not true, we should get the index of the row corresponding to that input
            let idx_vars= to_bit_sequence_le(*input as usize, Relu::num_vars()).map(|b| F::from(b as u64)).collect_vec();
            let input_field = input_mle.evaluate(&idx_vars);
            assert_eq!(input_field,F::from(*input));
            let output_field = output_mle.evaluate(&idx_vars);
            assert_eq!(output_field,F::from(*output));
        }
        //assert_eq!(expected,given);
    }
}