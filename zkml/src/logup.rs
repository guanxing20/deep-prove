//! Contains code for LogUp proving using GKR see: https://eprint.iacr.org/2023/1284.pdf for more.

use ff::Field;
use ff_ext::ExtensionField;
use gkr::{
    error::GKRError,
    structs::{Circuit, CircuitWitness, GKRInputClaims, IOPProof, IOPProverState, PointAndEval},
};
use itertools::izip;
use multilinear_extensions::mle::{DenseMultilinearExtension, FieldType, MultilinearExtension};
use rayon::prelude::*;
use simple_frontend::structs::{CellId, CircuitBuilder, ExtCellId};
use std::collections::HashMap;

use transcript::Transcript;

/// Function that builds the LogUp GKR circuit, the input `num_vars` is the number of variables the `p` and `q` MLEs will have.
pub fn logup_circuit<E: ExtensionField>(num_vars: usize, no_table_columns: usize) -> Circuit<E> {
    let cb = &mut CircuitBuilder::default();

    // For each column of the table and lookup wire we have an input witness
    let table_columns = (0..no_table_columns)
        .map(|_| cb.create_witness_in(1 << num_vars).1)
        .collect::<Vec<Vec<CellId>>>();
    let lookup_wire_columns = (0..no_table_columns)
        .map(|_| cb.create_witness_in(1 << num_vars).1)
        .collect::<Vec<Vec<CellId>>>();

    let multiplicity_poly = cb.create_witness_in(1 << num_vars).1;

    let minus_one_const = cb.create_constant_in(1 << num_vars, -1);

    let (table, lookup) = (
        cb.create_ext_cells(1 << num_vars),
        cb.create_ext_cells(1 << num_vars),
    );

    izip!(table.iter(), lookup.iter())
        .enumerate()
        .for_each(|(i, (table_cell, lookup_cell))| {
            let table_row = table_columns
                .iter()
                .map(|col| col[i])
                .collect::<Vec<CellId>>();
            let lookup_row = lookup_wire_columns
                .iter()
                .map(|col| col[i])
                .collect::<Vec<CellId>>();

            // Produce the merged table row
            cb.rlc(table_cell, &table_row, 0);

            // Produce the merged lookup row
            cb.rlc(lookup_cell, &lookup_row, 0);
        });

    let p_base = [multiplicity_poly, minus_one_const].concat();
    let q_start = [table, lookup].concat();

    // at each level take adjacent pairs (p0, q0) and (p1, q1) and compute the next levels p and q as
    // p_next = p0 * q1 + p1 * q0
    // q_next = q0 * q1
    // We do this because we don't want to perform costly field inversions at each step and this emulates
    // p0/q0 + p1/q1 which is equal to (p0q1 + p1q0)/q0q1
    //
    // At the first level it is slightly different because one set of evals is basefield and the other is extension field

    let (mut p, mut q): (Vec<ExtCellId<E>>, Vec<ExtCellId<E>>) = p_base
        .chunks(2)
        .zip(q_start.chunks(2))
        .map(|(p_is, q_is)| {
            let p_out = cb.create_ext_cell();
            cb.mul_ext_base(&p_out, &q_is[1], p_is[0], E::BaseField::ONE);

            cb.mul_ext_base(&p_out, &q_is[0], p_is[1], E::BaseField::ONE);

            let q_out = cb.create_ext_cell();
            cb.mul2_ext(&q_out, &q_is[0], &q_is[1], E::BaseField::ONE);
            (p_out, q_out)
        })
        .unzip();

    while p.len() > 1 && q.len() > 1 {
        (p, q) = p
            .chunks(2)
            .zip(q.chunks(2))
            .map(|(p_is, q_is)| {
                let p_out = cb.create_ext_cell();
                cb.mul2_ext(&p_out, &p_is[0], &q_is[1], E::BaseField::ONE);

                cb.mul2_ext(&p_out, &p_is[1], &q_is[0], E::BaseField::ONE);

                let q_out = cb.create_ext_cell();
                cb.mul2_ext(&q_out, &q_is[0], &q_is[1], E::BaseField::ONE);
                (p_out, q_out)
            })
            .unzip();
    }
    // Once the loop has finished we should be left with only one p and q,
    // the value stored in p should be the numerator of Sum p_original/q_original and should be equal to 0
    // the value stored in q is the product of all the evaluations of the input q mle and should be enforced to be non-zero by the verifier.
    assert_eq!(p.len(), 1);
    assert_eq!(q.len(), 1);

    p[0].cells.iter().for_each(|cell| cb.assert_const(*cell, 0));
    cb.create_witness_out_from_exts(&q);

    cb.configure();

    Circuit::new(cb)
}

/// Given MLEs `p(X)` and `q(X)` this function produces an [`IOPProof`] that
/// `SUM P(X)/Q(X) == 0`.
pub fn prove_logup<E: ExtensionField, T: Transcript<E>>(
    table: &[DenseMultilinearExtension<E>],
    lookups: &[DenseMultilinearExtension<E>],
    circuit: &Circuit<E>,
    transcript: &mut T,
) -> Option<(IOPProof<E>, E)> {
    // Check we have the same number of columns for each
    assert_eq!(table.len(), lookups.len());

    let num_vars = table[0].num_vars();

    // We need to squeeze one challenge here for merging the table and lookup columns
    let challenges = std::iter::repeat_with(|| {
        transcript
            .get_and_append_challenge(b"logup_challenge")
            .elements
    })
    .take(1)
    .collect::<Vec<E>>();

    // Convert the provided MLEs into their evaluations, they should all be basefield elements
    let table_evals = table
        .iter()
        .map(|col| match col.evaluations() {
            FieldType::Base(inner) => inner.clone(),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();

    let lookup_evals = lookups
        .iter()
        .map(|col| match col.evaluations() {
            FieldType::Base(inner) => inner.clone(),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();

    let challenge_powers = std::iter::successors(Some(E::ONE), |prev| Some(*prev * challenges[0]))
        .take(table.len() + 1)
        .collect::<Vec<E>>();
    // Compute the merged evaluations (to be used in calculating the multiplicity polynomial)
    let (merged_table, merged_lookup): (Vec<E>, Vec<E>) = (0..1 << num_vars)
        .into_par_iter()
        .map(|i| {
            let (table_entry, lookup_entry) = table_evals
                .iter()
                .zip(lookup_evals.iter())
                .enumerate()
                .fold(
                    (challenge_powers[table.len()], challenge_powers[table.len()]),
                    |(table_acc, lookup_acc), (j, (table_col, lookup_col))| {
                        let out_table_acc = table_acc + challenge_powers[j] * table_col[i];
                        let out_lookup_acc = lookup_acc + challenge_powers[j] * lookup_col[i];

                        (out_table_acc, out_lookup_acc)
                    },
                );

            (table_entry, lookup_entry)
        })
        .unzip();

    let multiplicity_poly = compute_multiplicity_poly(&merged_table, &merged_lookup);

    // We calculate the product of the evaluations of the combined denominator here as it is an output of the GKR circuit
    let denom_prod = merged_table
        .iter()
        .chain(merged_lookup.iter())
        .product::<E>();
    let denom_prod_mle =
        DenseMultilinearExtension::from_evaluations_slice(1, denom_prod.as_bases());
    // Produce a vector of all the witness inputs
    let wits_in = table
        .iter()
        .chain(lookups.iter())
        .cloned()
        .chain([multiplicity_poly])
        .collect::<Vec<DenseMultilinearExtension<E>>>();

    let mut witness = CircuitWitness::new(circuit, challenges);
    witness.add_instance(circuit, wits_in);

    // Squeeze a challenge to be used in evaluating the output mle (this is denom_prod flattened ot basefield elements)
    let output_point = std::iter::repeat_with(|| {
        transcript
            .get_and_append_challenge(b"output_challenge")
            .elements
    })
    .take(1)
    .collect::<Vec<E>>();
    // Compute the output eval
    let output_eval = denom_prod_mle.evaluate(&output_point);

    let (proof, _) = IOPProverState::prove_parallel(
        circuit,
        &witness,
        vec![],
        vec![PointAndEval::new(output_point, output_eval)],
        1,
        transcript,
    );

    Some((proof, denom_prod))
}

/// Verifies a GKR proof that `SUM M(X)/(a + T(X)) - 1/(a + L(X)) == 0` when provided with `PROD Q(X)`.
/// It also errors if `PROD (a + T(X))(a + L(X)) == 0`
pub fn verify_logup<E: ExtensionField, T: Transcript<E>>(
    denom_prod: E,
    proof: IOPProof<E>,
    circuit: &Circuit<E>,
    transcript: &mut T,
) -> Result<GKRInputClaims<E>, GKRError> {
    if denom_prod.is_zero().into() {
        return Err(GKRError::VerifyError(
            "The product of the denominator was zero so proof is invalid",
        ));
    }

    let challenges = std::iter::repeat_with(|| {
        transcript
            .get_and_append_challenge(b"logup_challenge")
            .elements
    })
    .take(1)
    .collect::<Vec<E>>();

    let output_point = std::iter::repeat_with(|| {
        transcript
            .get_and_append_challenge(b"output_challenge")
            .elements
    })
    .take(1)
    .collect::<Vec<E>>();
    let denom_prod_mle =
        DenseMultilinearExtension::from_evaluations_slice(1, denom_prod.as_bases());
    let denom_prod_eval = denom_prod_mle.evaluate(&output_point);

    gkr::structs::IOPVerifierState::verify_parallel(
        circuit,
        &challenges,
        vec![],
        vec![PointAndEval::new(output_point, denom_prod_eval)],
        proof,
        0,
        transcript,
    )
}

/// Function that when provided with the merged table and merged lookups computes the multiplicity polynomial
fn compute_multiplicity_poly<E: ExtensionField>(
    merged_table: &[E],
    merged_lookups: &[E],
) -> DenseMultilinearExtension<E> {
    // Create HashMaps to keep track of the number of entries
    let mut h_lookup = HashMap::new();
    let mut h_table = HashMap::new();
    let num_vars = merged_table.len().ilog2() as usize;

    // For each value in the merged table and merged lookups create an entry in the respective HashMap if its not already present
    // otherwise simply increment the count for how many times we've seeen this element
    merged_table
        .iter()
        .zip(merged_lookups.iter())
        .for_each(|(&table_entry, &lookup_entry)| {
            *h_table
                .entry(table_entry)
                .or_insert_with(|| E::BaseField::ZERO) += E::BaseField::ONE;
            *h_lookup
                .entry(lookup_entry)
                .or_insert_with(|| E::BaseField::ZERO) += E::BaseField::ONE;
        });

    // Calculate multiplicity polynomial evals, these are calculated as (no. times looked up) / (no. times in table)
    // If a value is present in the table but is not looked up we set its multiplicity to be 0.
    let multiplicity_evals = merged_table
        .iter()
        .map(|value| {
            if let Some(lookup_count) = h_lookup.get(value) {
                *lookup_count * h_table.get(value).unwrap().invert().unwrap()
            } else {
                E::BaseField::ZERO
            }
        })
        .collect::<Vec<E::BaseField>>();

    DenseMultilinearExtension::from_evaluations_vec(num_vars, multiplicity_evals)
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        Rng, RngCore, SeedableRng,
        rngs::{OsRng, StdRng},
    };
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use transcript::BasicTranscript;

    use super::*;

    #[test]
    fn test_logup_gkr() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
        for n in 4..20 {
            println!("Testing with {n} number of variables");
            // Make the circuit for n variables with two columns
            let circuit = logup_circuit::<GoldilocksExt2>(n, 2);

            // Make two columns for the table
            let table_column_1 = (0..1 << n)
                .map(|_| {
                    let random: u64 = rng.gen::<u64>() + 1u64;
                    Goldilocks::from(random)
                })
                .collect::<Vec<_>>();

            let table_column_2 = (0..1 << n)
                .map(|_| {
                    let random: u64 = rng.gen::<u64>() + 1u64;
                    Goldilocks::from(random)
                })
                .collect::<Vec<_>>();

            // Make two lookup columns, here we reverse the order of the original table columns, that way each
            // value is looked up but not in the same order
            let mut lookup_column_1 = table_column_1.clone();
            lookup_column_1.reverse();

            let mut lookup_column_2 = table_column_2.clone();
            lookup_column_2.reverse();

            let table = [table_column_1, table_column_2]
                .map(|col| DenseMultilinearExtension::from_evaluations_vec(n, col));
            let lookups = [lookup_column_1, lookup_column_2]
                .map(|col| DenseMultilinearExtension::from_evaluations_vec(n, col));

            // Initiate a new transcript for the prover
            let mut prover_transcript = BasicTranscript::<GoldilocksExt2>::new(b"test");
            // Make the proof and the claimed product of the denominator polynomial
            let now = std::time::Instant::now();
            let (proof, denom_prod) =
                prove_logup(&table, &lookups, &circuit, &mut prover_transcript).unwrap();
            println!("Total time to run prove function: {:?}", now.elapsed());

            // Make a transcript for the verifier
            let mut verifier_transcript = BasicTranscript::<GoldilocksExt2>::new(b"test");
            // Generate the verifiers claim that they need to check against the original input polynomials
            let claim = verify_logup(denom_prod, proof, &circuit, &mut verifier_transcript);
            assert!(claim.is_ok());

            let input_claims = claim.unwrap();

            // For each input polynomial we check that it evaluates to the value output in GKRClaim at the point
            // in GKRClaim.
            for (input_poly, point_and_eval) in table
                .iter()
                .chain(lookups.iter())
                .zip(input_claims.point_and_evals.iter())
            {
                let actual_eval = input_poly.evaluate(&point_and_eval.point);

                assert_eq!(actual_eval, point_and_eval.eval);
            }

            let expected_m_poly =
                DenseMultilinearExtension::from_evaluations_vec(n, vec![Goldilocks::ONE; 1 << n]);
            let final_point_and_eval = input_claims.point_and_evals.last().unwrap();

            let actual_eval = expected_m_poly.evaluate(&final_point_and_eval.point);

            assert_eq!(actual_eval, final_point_and_eval.eval);
        }
    }
}
