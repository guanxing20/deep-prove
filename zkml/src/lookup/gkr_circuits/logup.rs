//! Contains code for LogUp proving using GKR see: https://eprint.iacr.org/2023/1284.pdf for more.

use ff::Field;
use ff_ext::ExtensionField;
use gkr::{
    error::GKRError,
    structs::{Circuit, CircuitWitness, GKRInputClaims, IOPProof, IOPProverState, PointAndEval},
};
use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
use simple_frontend::structs::{CellId, CircuitBuilder, ExtCellId};

use crate::lookup::gkr_circuits::add_fractions;

use super::super::utils::{compute_multiplicity_poly, merge_columns};
use transcript::Transcript;
/// Function that builds the LogUp GKR circuit, the input `num_vars` is the number of variables the `p` and `q` MLEs will have.
pub fn logup_circuit<E: ExtensionField>(
    table_num_vars: usize,
    lookup_num_vars: usize,
    no_table_columns: usize,
) -> Circuit<E> {
    let cb = &mut CircuitBuilder::default();

    // For each column of the table and lookup wire we have an input witness
    let table_columns = (0..no_table_columns)
        .map(|_| cb.create_witness_in(1 << table_num_vars).1)
        .collect::<Vec<Vec<CellId>>>();
    let lookup_wire_columns = (0..no_table_columns)
        .map(|_| cb.create_witness_in(1 << lookup_num_vars).1)
        .collect::<Vec<Vec<CellId>>>();

    let multiplicity_poly = cb.create_witness_in(1 << table_num_vars).1;

    let (table, lookup) = (
        cb.create_ext_cells(1 << table_num_vars),
        cb.create_ext_cells(1 << lookup_num_vars),
    );

    table.iter().enumerate().for_each(|(i, table_cell)| {
        let table_row = table_columns
            .iter()
            .map(|col| col[i])
            .collect::<Vec<CellId>>();

        // Produce the merged table row
        cb.rlc(table_cell, &table_row, 0);
    });

    lookup.iter().enumerate().for_each(|(i, lookup_cell)| {
        let lookup_row = lookup_wire_columns
            .iter()
            .map(|col| col[i])
            .collect::<Vec<CellId>>();

        // Produce the merged lookup row
        cb.rlc(lookup_cell, &lookup_row, 0);
    });

    // at each level take adjacent pairs (p0, q0) and (p1, q1) and compute the next levels p and q as
    // p_next = p0 * q1 + p1 * q0
    // q_next = q0 * q1
    // We do this because we don't want to perform costly field inversions at each step and this emulates
    // p0/q0 + p1/q1 which is equal to (p0q1 + p1q0)/q0q1
    //
    // At the first level it is slightly different because one set of evals is basefield and the other is extension field
    // We do the first round of summing for the table and the lookups seperately since the lookups don't requie an input for the numerator
    let (mut table_num, mut table_denom): (Vec<ExtCellId<E>>, Vec<ExtCellId<E>>) =
        multiplicity_poly
            .chunks(2)
            .zip(table.chunks(2))
            .map(|(nums, denoms)| {
                let num_out = cb.create_ext_cell();
                cb.mul_ext_base(&num_out, &denoms[1], nums[0], E::BaseField::ONE);

                cb.mul_ext_base(&num_out, &denoms[0], nums[1], E::BaseField::ONE);

                let denom_out = cb.create_ext_cell();
                cb.mul2_ext(&denom_out, &denoms[0], &denoms[1], E::BaseField::ONE);
                (num_out, denom_out)
            })
            .unzip();

    let (mut lookup_num, mut lookup_denom): (Vec<ExtCellId<E>>, Vec<ExtCellId<E>>) = lookup
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

    while table_num.len() > 1 && table_denom.len() > 1 {
        (table_num, table_denom) = add_fractions(cb, &table_num, &table_denom);
    }

    while lookup_num.len() > 1 && lookup_denom.len() > 1 {
        (lookup_num, lookup_denom) = add_fractions(cb, &lookup_num, &lookup_denom);
    }
    // Once the loop has finished we should be left with only one p and q,
    // the value stored in p should be the numerator of Sum p_original/q_original and should be equal to 0
    // the value stored in q is the product of all the evaluations of the input q mle and should be enforced to be non-zero by the verifier.
    assert_eq!(table_num.len(), 1);
    assert_eq!(table_denom.len(), 1);
    assert_eq!(lookup_num.len(), 1);
    assert_eq!(lookup_denom.len(), 1);

    let numerators = [table_num, lookup_num].concat();
    let denominators = [table_denom, lookup_denom].concat();

    let (final_numerator, final_denominator) = add_fractions(cb, &numerators, &denominators);
    final_numerator[0]
        .cells
        .iter()
        .for_each(|cell| cb.assert_const(*cell, 0));
    cb.create_witness_out_from_exts(&final_denominator);

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
) -> Option<(IOPProof<E>, E, DenseMultilinearExtension<E>)> {
    // Check we have the same number of columns for each
    assert_eq!(table.len(), lookups.len());

    // We need to squeeze one challenge here for merging the table and lookup columns
    let challenges = std::iter::repeat_with(|| {
        transcript
            .get_and_append_challenge(b"logup_challenge")
            .elements
    })
    .take(1)
    .collect::<Vec<E>>();

    // Compute the merged table and lookup
    let merged_table = merge_columns(table, challenges[0]);
    let merged_lookup = merge_columns(lookups, challenges[0]);

    // Comput the multiplicity polynomial
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
        .chain([&multiplicity_poly])
        .cloned()
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

    Some((proof, denom_prod, multiplicity_poly))
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

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        Rng, RngCore, SeedableRng,
        rngs::{OsRng, StdRng},
    };
    use gkr::utils::i64_to_field;
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use transcript::BasicTranscript;

    use super::*;

    #[test]
    fn test_logup_gkr() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());

        // We make two fixed table columns for the test
        let subtractor = 1i64 << 15;
        let (in_column, out_column): (Vec<Goldilocks>, Vec<Goldilocks>) = (0i64..1 << 16)
            .map(|i| {
                let shifted = i - subtractor;
                let cell_1 = shifted;
                let cell_2 = if shifted <= 0 { 0 } else { shifted };

                (
                    i64_to_field::<Goldilocks>(cell_1),
                    i64_to_field::<Goldilocks>(cell_2),
                )
            })
            .unzip();

        let table = [in_column.as_slice(), out_column.as_slice()]
            .map(|col| DenseMultilinearExtension::from_evaluations_slice(16, col));

        let random_combiner = GoldilocksExt2::random(&mut rng);
        let merged_table = merge_columns(&table, random_combiner);

        for n in 4..20 {
            println!("Testing with {n} number of variables");
            // First make two random columns that should pass
            let (lookup_in, lookup_out): (Vec<Goldilocks>, Vec<Goldilocks>) = (0..1 << n)
                .map(|_| {
                    let position = rng.gen_range(0usize..1 << 16);
                    (in_column[position], out_column[position])
                })
                .unzip();

            let lookups = [lookup_in.as_slice(), lookup_out.as_slice()]
                .map(|col| DenseMultilinearExtension::from_evaluations_slice(n, col));
            let merged_lookups = merge_columns(&lookups, random_combiner);

            let expected_multiplicity_poly =
                compute_multiplicity_poly(&merged_table, &merged_lookups);

            // Make the circuit for n variables with two columns
            let circuit = logup_circuit::<GoldilocksExt2>(16, n, 2);

            // Initiate a new transcript for the prover
            let mut prover_transcript = BasicTranscript::<GoldilocksExt2>::new(b"test");
            // Make the proof and the claimed product of the denominator polynomial
            let now = std::time::Instant::now();
            let (proof, denom_prod, _) =
                prove_logup(&table, &lookups, &circuit, &mut prover_transcript).unwrap();
            println!("Total time to run prove function: {:?}", now.elapsed());

            // Make a transcript for the verifier
            let mut verifier_transcript = BasicTranscript::<GoldilocksExt2>::new(b"test");
            // Generate the verifiers claim that they need to check against the original input polynomials
            let claim = verify_logup(
                denom_prod,
                proof.clone(),
                &circuit,
                &mut verifier_transcript,
            );
            assert!(claim.is_ok());

            let input_claims = claim.unwrap();

            // For each input polynomial we check that it evaluates to the value output in GKRClaim at the point
            // in GKRClaim.
            for (input_poly, point_and_eval) in table
                .iter()
                .chain(lookups.iter())
                .chain(std::iter::once(&expected_multiplicity_poly))
                .zip(input_claims.point_and_evals.iter())
            {
                let actual_eval = input_poly.evaluate(&point_and_eval.point);
                assert_eq!(actual_eval, point_and_eval.eval);
            }
        }
    }
}
