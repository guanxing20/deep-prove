#![feature(iter_next_chunk)]

use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::mle::DenseMultilinearExtension;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator,  ParallelIterator};
use serde::{Deserialize, Serialize};
use transcript::{BasicTranscript, Transcript};
mod commit;
mod matrix;
mod model;
mod onnx_parse;
mod lookup;
mod testing;
mod activation;
mod iop;

/// Claim type to accumulate in this protocol, for a certain polynomial, known in the context.
/// f(point) = eval
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Claim<E> {
    point: Vec<E>,
    eval: E,
}

impl<E> Claim<E> {
    pub fn from(point: Vec<E>,eval:E) -> Self {
        Self {
            point,
            eval,
        }
    }
}

/// Element is u64 right now to withstand the overflow arithmetics when running inference for any kinds of small models.
/// With quantization this is not needed anymore and we can try changing back to u16 or u32 but perf gains should be minimal.
type Element = u64;

/// Returns the default transcript the prover and verifier must instantiate to validate a proof.
pub fn default_transcript<E: ExtensionField>() -> BasicTranscript<E> {
    BasicTranscript::new(b"m2vec")
}

pub fn vector_to_field_par<E: ExtensionField>(v: &[Element]) -> Vec<E> {
    v.par_iter().map(|v| E::from(*v as u64)).collect::<Vec<_>>()
}
pub fn vector_to_field_par_into<E: ExtensionField>(v: Vec<Element>) -> Vec<E> {
    v.into_par_iter().map(|v| E::from(v as u64)).collect::<Vec<_>>()
}

pub fn pad_vector<E: ExtensionField>(mut v: Vec<E>) -> Vec<E> {
    if !v.len().is_power_of_two() {
        v.resize(v.len().next_power_of_two(), E::ZERO);
    }
    v
}
/// Returns a MLE out of the given vector, of the right length
// TODO : make that part of tensor somehow?
pub(crate) fn vector_to_mle<E: ExtensionField>(v: Vec<E>) -> DenseMultilinearExtension<E> {
    let v = pad_vector(v);
    DenseMultilinearExtension::from_evaluation_vec_smart(v.len().ilog2() as usize, v)
}

/// Returns the bit sequence of num of bit_length length.
pub(crate) fn to_bit_sequence_le(num: usize, bit_length: usize) -> impl DoubleEndedIterator<Item = usize> {
    assert!(
        bit_length as u32 <= usize::BITS,
        "bit_length cannot exceed usize::BITS"
    );
    (0..bit_length).map(move |i| ((num >> i) & 1) as usize)
}

pub trait VectorTranscript<E: ExtensionField> {
    fn read_challenges(&mut self, n: usize) -> Vec<E>;
}

#[cfg(not(test))]
impl<T: Transcript<E>, E: ExtensionField> VectorTranscript<E> for T {
    fn read_challenges(&mut self, n: usize) -> Vec<E> {
        (0..n).map(|_| self.read_challenge().elements).collect_vec()
    }
}

#[cfg(test)]
impl<T: Transcript<E>, E: ExtensionField> VectorTranscript<E> for T {
    fn read_challenges(&mut self, n: usize) -> Vec<E> {
        (0..n).map(|_| E::ONE).collect_vec()
    }
}

#[cfg(test)]
mod test {
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::MultilinearExtension;

    use crate::{default_transcript, iop::{prover::Prover, verifier::{verify, IO}, Context}, onnx_parse::load_mlp, testing::random_vector, to_bit_sequence_le, vector_to_field_par, vector_to_mle, Element};
    use ff_ext::ff::Field;

    type E = GoldilocksExt2;

    #[test]
    fn test_model_run() {
        let filepath = "assets/model.onnx";
        let model = load_mlp::<Element>(&filepath).unwrap();
        println!("[+] Loaded onnx file");
        let ctx = Context::<E>::generate(&model).expect("unable to generate context");
        println!("[+] Setup parameters");

        let shape = model.input_shape();
        assert_eq!(shape.len(),1);
        let input = random_vector(shape[0]);

        let trace = model.run(input.clone());
        let output = trace.final_output().to_vec();
        println!("[+] Run inference. Result: {:?}",output); 

        let mut prover_transcript = default_transcript();
        let prover = Prover::new(&ctx,&mut prover_transcript);
        println!("[+] Run prover");
        let proof = prover.prove(trace).expect("unable to generate proof");

        let mut verifier_transcript = default_transcript();
        let io = IO::new(vector_to_field_par(&input), output.to_vec());
        verify(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
        println!("[+] Verify proof: valid");
    }


    // TODO: move below code to a vector module

    #[test]
    fn test_vector_mle() {
        let n = 10;
        let v = (0..n).map(|_| E::random(&mut thread_rng())).collect_vec();
        let mle = vector_to_mle(v.clone());
        let random_index = thread_rng().gen_range(0..v.len());
        let eval = to_bit_sequence_le(random_index, v.len().next_power_of_two().ilog2() as usize)
            .map(|b| E::from(b as u64))
            .collect_vec();
        let output = mle.evaluate(&eval);
        assert_eq!(output, v[random_index]);
    }
}
