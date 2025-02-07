#![feature(iter_next_chunk)]

use derive_more::From;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
use transcript::Transcript;
mod matrix;
mod model;
mod prover;

// TODO: use a real tensor
#[derive(Debug, Clone, From)]
struct Tensor<E>(Vec<E>);

/// Returns a MLE out of the given vector, of the right length
// TODO : make that part of tensor somehow?
pub(crate) fn vector_to_mle<E: ExtensionField>(mut v: Vec<E>) -> DenseMultilinearExtension<E> {
    v.resize(v.len().next_power_of_two(), E::ZERO);
    DenseMultilinearExtension::from_evaluation_vec_smart(v.len().ilog2() as usize, v)
}

/// Returns the bit sequence of num of bit_length length.
pub(crate) fn to_bit_sequence_le(num: usize, bit_length: usize) -> impl Iterator<Item = usize> {
    assert!(
        bit_length as u32 <= usize::BITS,
        "bit_length cannot exceed usize::BITS"
    );
    (0..bit_length).map(move |i| ((num >> i) & 1) as usize)
}

pub trait VectorTranscript<E: ExtensionField> {
    fn read_challenges(&mut self, n: usize) -> Vec<E>;
}

impl<T: Transcript<E>, E: ExtensionField> VectorTranscript<E> for T {
    fn read_challenges(&mut self, n: usize) -> Vec<E> {
        (0..n).map(|_| self.read_challenge().elements).collect_vec()
    }
}

#[cfg(test)]
mod test {
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};

    use crate::{to_bit_sequence_le, vector_to_mle};
    use ff_ext::ff::Field;

    type E = GoldilocksExt2;

    #[test]
    fn test_vector_mle() {
        let n = 10;
        let mut v = (0..n).map(|_| E::random(&mut thread_rng())).collect_vec();
        let mle = vector_to_mle(v.clone());
        let random_index = thread_rng().gen_range(0..v.len());
        let eval = to_bit_sequence_le(random_index, v.len().next_power_of_two().ilog2() as usize)
            .map(|b| E::from(b as u64))
            .collect_vec();
        let output = mle.evaluate(&eval);
        assert_eq!(output, v[random_index]);
    }
}
