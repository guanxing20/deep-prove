//! Contains code for building a GKR to obtain the output of a fractional sumcheck for lookup columns when using the LogUp protocol.

use ff::Field;
use ff_ext::ExtensionField;
use gkr::structs::Circuit;

use simple_frontend::structs::{CellId, CircuitBuilder, ExtCellId};

use super::add_fractions;

pub fn lookup_wire_fractional_sumcheck<E: ExtensionField>(
    no_lookup_columns: usize,
    num_vars: usize,
) -> Circuit<E> {
    let cb = &mut CircuitBuilder::default();

    // For each column of the table and lookup wire we have an input witness
    let lookup_columns = (0..no_lookup_columns)
        .map(|_| cb.create_witness_in(1 << num_vars).1)
        .collect::<Vec<Vec<CellId>>>();
    let lookups = cb.create_ext_cells(1 << num_vars);

    lookups.iter().enumerate().for_each(|(i, lookup_cell)| {
        let lookup_row = lookup_columns
            .iter()
            .map(|col| col[i])
            .collect::<Vec<CellId>>();
        cb.combine_columns(lookup_cell, &lookup_row, 0, 1);
    });

    let (mut lookup_nums, mut lookup_denoms): (Vec<ExtCellId<E>>, Vec<ExtCellId<E>>) = lookups
        .chunks(2)
        .map(|denoms| {
            let num_out = cb.create_ext_cell();
            cb.add_ext(&num_out, &denoms[0], -E::BaseField::ONE);
            cb.add_ext(&num_out, &denoms[1], -E::BaseField::ONE);

            let denom_out = cb.create_ext_cell();
            cb.mul2_ext(&denom_out, &denoms[0], &denoms[1], E::BaseField::ONE);
            (num_out, denom_out)
        })
        .unzip();

    while lookup_nums.len() > 1 && lookup_denoms.len() > 1 {
        (lookup_nums, lookup_denoms) = add_fractions(cb, &lookup_nums, &lookup_denoms);
    }

    lookup_nums
        .into_iter()
        .zip(lookup_denoms.into_iter())
        .for_each(|(num, denom)| {
            cb.create_witness_out_from_exts(&[num]);
            cb.create_witness_out_from_exts(&[denom]);
        });

    cb.configure();

    Circuit::new(cb)
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        Rng, RngCore, SeedableRng,
        rngs::{OsRng, StdRng},
    };
    use gkr::{
        error::GKRError,
        structs::{CircuitWitness, GKRInputClaims, IOPProof, IOPProverState, PointAndEval},
    };
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
    use transcript::{BasicTranscript, Transcript};

    use super::*;

    #[test]
    fn test_fractional_sumcheck() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());

        let challenge = GoldilocksExt2::random(&mut rng);
        let const_challenge = GoldilocksExt2::random(&mut rng);
        let challenges = vec![challenge, const_challenge];
        for n in 4..20 {
            println!("Testing with {n} number of variables");
            // First make two random columns that should pass
            let lookup_column = (0..1 << n)
                .map(|_| Goldilocks::random(&mut rng))
                .collect::<Vec<Goldilocks>>();

            let piop_denoms = lookup_column
                .iter()
                .map(|val| const_challenge + challenge * GoldilocksExt2::from(*val))
                .collect::<Vec<GoldilocksExt2>>();
            let first_denom = piop_denoms[0];
            let (expected_num, expected_denom) = piop_denoms.iter().skip(1).fold(
                (-GoldilocksExt2::ONE, first_denom),
                |(acc_num, acc_denom), &denom_i| {
                    (acc_num * denom_i - acc_denom, acc_denom * denom_i)
                },
            );
            // Make the circuit for n variables with one columns
            let circuit = lookup_wire_fractional_sumcheck::<GoldilocksExt2>(1, n);

            let lookup_mle = DenseMultilinearExtension::from_evaluations_slice(n, &lookup_column);
            // Initiate a new transcript for the prover
            let mut prover_transcript = BasicTranscript::<GoldilocksExt2>::new(b"test");
            // Make the proof and the claimed product of the denominator polynomial
            let now = std::time::Instant::now();
            let (proof, numerators, denominators) = prove_fractional_sumcheck(
                &[lookup_mle.clone()],
                &circuit,
                challenges.clone(),
                &mut prover_transcript,
            )
            .unwrap();
            println!("Total time to run prove function: {:?}", now.elapsed());

            assert_eq!(numerators[0], expected_num);
            assert_eq!(denominators[0], expected_denom);

            // Initiate a new transcript for the verifier
            let mut verifier_transcript = BasicTranscript::<GoldilocksExt2>::new(b"test");
            let claims = verify_fractional_sumcheck(
                proof,
                &numerators,
                &denominators,
                &challenges,
                &circuit,
                &mut verifier_transcript,
            )
            .unwrap();

            let p_and_e = claims.point_and_evals[0].clone();

            assert_eq!(p_and_e.eval, lookup_mle.evaluate(&p_and_e.point));
        }
    }

    fn prove_fractional_sumcheck<E: ExtensionField, T: Transcript<E>>(
        lookups: &[DenseMultilinearExtension<E>],
        circuit: &Circuit<E>,
        challenges: Vec<E>,
        transcript: &mut T,
    ) -> Option<(IOPProof<E>, Vec<E>, Vec<E>)> {
        let mut witness = CircuitWitness::new(circuit, challenges);
        witness.add_instance(circuit, lookups.to_vec());

        // Squeeze a challenge to be used in evaluating the output mle (this is denom_prod flattened ot basefield elements)
        let output_point = std::iter::repeat_with(|| {
            transcript
                .get_and_append_challenge(b"output_challenge")
                .elements
        })
        .take(1)
        .collect::<Vec<E>>();
        // Compute the output eval
        let witness_out_evals = witness
            .witness_out_ref()
            .iter()
            .map(|mle| PointAndEval::<E>::new(output_point.clone(), mle.evaluate(&output_point)))
            .collect::<Vec<PointAndEval<E>>>();

        let (proof, _) = IOPProverState::prove_parallel(
            circuit,
            &witness,
            vec![],
            witness_out_evals,
            1,
            transcript,
        );

        let (numerators, denominators): (Vec<E>, Vec<E>) = witness
            .witness_out_ref()
            .chunks(2)
            .map(|chunk| {
                let num = E::from_bases(chunk[0].get_base_field_vec());
                let denom = E::from_bases(chunk[1].get_base_field_vec());
                (num, denom)
            })
            .unzip();

        Some((proof, numerators, denominators))
    }

    pub fn verify_fractional_sumcheck<E: ExtensionField, T: Transcript<E>>(
        proof: IOPProof<E>,
        numerators: &[E],
        denominators: &[E],
        challenges: &[E],
        circuit: &Circuit<E>,
        transcript: &mut T,
    ) -> Result<GKRInputClaims<E>, GKRError> {
        let output_point = std::iter::repeat_with(|| {
            transcript
                .get_and_append_challenge(b"output_challenge")
                .elements
        })
        .take(1)
        .collect::<Vec<E>>();

        let point = output_point[0];

        let witness_out_evals = numerators
            .iter()
            .zip(denominators.iter())
            .flat_map(|(&num, &denom)| {
                let num_bases = num.as_bases();
                let denom_bases = denom.as_bases();

                let num_eval =
                    E::from((num_bases[1] - num_bases[0])) * point + E::from(num_bases[0]);
                let denom_eval =
                    E::from((denom_bases[1] - denom_bases[0])) * point + E::from(denom_bases[0]);
                vec![
                    PointAndEval::<E>::new(output_point.clone(), num_eval),
                    PointAndEval::<E>::new(output_point.clone(), denom_eval),
                ]
            })
            .collect::<Vec<PointAndEval<E>>>();

        gkr::structs::IOPVerifierState::verify_parallel(
            circuit,
            challenges,
            vec![],
            witness_out_evals,
            proof,
            0,
            transcript,
        )
    }
}
