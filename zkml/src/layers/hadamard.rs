//! This module implements the proving and verifying of a hadamard product
//! as v[i] = v1[i] * v2[i]
//! It can not be integrated as a single layer in the model, but rather it's a
//! a module that can be used within other layers. FOr example it is used within
//! convolution.

use ff_ext::ExtensionField;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::VirtualPolynomial,
};
use sumcheck::structs::IOPProverState;
use transcript::Transcript;

use crate::{Claim, Element, Tensor, commit::compute_betas_eval};

// - v1, the two input vectors
// - v3 = v1 ° v2, the output vector
// - the last claim on the output vector v3(r) = y
// .  - remember the proving is done backwards
// - We want to prove that $\sum_i v_1(i) * v_2(i) * eq(r,i) = y = v_3(r)$
// - This output one point to evaluate on v1 and v2 and eq.
// NOTE: output claim MUST be valid on v3 but v3 is never computed on this method !
// In the conv application:
//   - v1 is the output of the conv
// . - v2 clearing garbage tensor. the verifier can evaluate v2 and eq easily
//   - the v1 claim is now passed to the previous layer (we prove in reverse)
pub fn prove<F: ExtensionField, T: Transcript<F>>(
    transcript: &mut T,
    output_claim: Claim<F>,
    v1: Tensor<Element>,
    v2: Tensor<Element>,
) -> Claim<F> {
    assert_eq!(
        output_claim.point.len(),
        v1.get_data().len().ilog2() as usize
    );
    assert_eq!(
        output_claim.point.len(),
        v2.get_data().len().ilog2() as usize
    );
    assert!(v1.get_shape().iter().all(|x| x.is_power_of_two()));
    assert!(v2.get_shape().iter().all(|x| x.is_power_of_two()));
    let beta_poly = compute_betas_eval(&output_claim.point).into_mle();
    let v1_mle = v1.to_mle_flat::<F>();
    let v2_mle = v2.to_mle_flat::<F>();
    let mut vp = VirtualPolynomial::<F>::new(v1_mle.num_vars());
    vp.add_mle_list(vec![v1_mle.into(), v2_mle.into(), beta_poly.into()], F::ONE);
    #[allow(deprecated)]
    let (proof, state) = IOPProverState::<F>::prove_parallel(vp, transcript);
    debug_assert!({
        let computed_eval = proof.extract_sum();
        let given_eval = output_claim.eval;
        computed_eval == given_eval
    });
    output_claim
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;
    use transcript::BasicTranscript;

    use super::*;
    use crate::{default_transcript, testing::random_field_vector};

    #[test]
    fn test_hadamard_proving() {
        let mut transcript = default_transcript();
        let n: usize = 10;
        let v1 = Tensor::random(vec![n]).pad_next_power_of_two();
        let v2 = Tensor::random(vec![n]).pad_next_power_of_two();
        let r = random_field_vector(n.next_power_of_two().ilog2() as usize);
        let expected_output = v1.mul(&v2);
        let output_mle = expected_output.to_mle_flat::<GoldilocksExt2>();
        let expected_eval = output_mle.evaluate(&r);
        let output_claim = Claim::new(r, expected_eval);
        let output_claim = prove(&mut transcript, output_claim, v1, v2);
        println!("output_claim: {:?}", output_claim);
    }
}
