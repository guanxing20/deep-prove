pub mod circuit;
pub mod error;
pub mod prover;
pub mod structs;
pub mod verifier;

#[cfg(test)]
mod tests {
    use ark_std::rand::thread_rng;
    use ff::Field;
    use ff_ext::ExtensionField;
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};

    use crate::{
        default_transcript,
        lookup::logup_gkr::{
            prover::batch_prove, structs::LogUpInput, verifier::verify_logup_proof,
        },
        quantization::Fieldizer,
        testing::random_vector,
    };

    #[test]
    fn test_logup_prove() {
        let mut rng = thread_rng();
        for n in 5..15 {
            let column = random_vector(1 << n)
                .into_iter()
                .map(|elem| {
                    let f: GoldilocksExt2 = elem.to_field();
                    f.as_bases()[0]
                })
                .collect::<Vec<Goldilocks>>();

            let column_2 = random_vector(1 << n)
                .into_iter()
                .map(|elem| {
                    let f: GoldilocksExt2 = elem.to_field();
                    f.as_bases()[0]
                })
                .collect::<Vec<Goldilocks>>();

            let constant_challenge = GoldilocksExt2::random(&mut rng);
            let column_separation_challenge = GoldilocksExt2::random(&mut rng);

            let column_evals = vec![column.clone(), column_2.clone()];
            let lookup_input = LogUpInput::<GoldilocksExt2>::new_lookup(
                column_evals.clone(),
                constant_challenge,
                column_separation_challenge,
                1,
            )
            .unwrap();

            let mut prover_transcript = default_transcript::<GoldilocksExt2>();
            let now = std::time::Instant::now();
            let proof = batch_prove(&lookup_input, &mut prover_transcript).unwrap();
            println!("Elapsed time proving: {:?}", now.elapsed());

            let mut verifier_transcript = default_transcript::<GoldilocksExt2>();
            let claims = verify_logup_proof(
                &proof,
                2,
                constant_challenge,
                column_separation_challenge,
                &mut verifier_transcript,
            )
            .unwrap();

            claims
                .claims()
                .iter()
                .zip(column_evals.into_iter())
                .for_each(|(claim, evaluations)| {
                    let mle = DenseMultilinearExtension::<GoldilocksExt2>::from_evaluations_vec(
                        n,
                        evaluations,
                    );
                    assert_eq!(claim.eval, mle.evaluate(&claim.point))
                });
        }
    }
}
