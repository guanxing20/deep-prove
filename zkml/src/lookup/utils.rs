use ff::Field;
use ff_ext::ExtensionField;
use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
use std::collections::HashMap;

/// Function that when provided with the merged table and merged lookups computes the multiplicity polynomial
pub fn compute_multiplicity_poly<E: ExtensionField>(
    merged_table: &[E],
    merged_lookups: &[E],
) -> DenseMultilinearExtension<E> {
    // Create HashMaps to keep track of the number of entries
    let mut h_lookup = HashMap::new();
    let mut h_table = HashMap::new();
    let num_vars = merged_table.len().ilog2() as usize;

    // For each value in the merged table and merged lookups create an entry in the respective HashMap if its not already present
    // otherwise simply increment the count for how many times we've seeen this element
    merged_table.iter().for_each(|&table_entry| {
        *h_table
            .entry(table_entry)
            .or_insert_with(|| E::BaseField::ZERO) += E::BaseField::ONE;
    });

    merged_lookups.iter().for_each(|&lookup_entry| {
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

/// Function that merges MLE columns of a table or lookups into a single vector of values
pub fn merge_columns<E: ExtensionField>(
    columns: &[DenseMultilinearExtension<E>],
    challenge: E,
) -> Vec<E> {
    // Convert the provided MLEs into their evaluations, they should all be basefield elements
    let column_evals = columns
        .iter()
        .map(|col| col.get_base_field_vec())
        .collect::<Vec<_>>();

    let num_vars = columns[0].num_vars();

    let challenge_powers = std::iter::successors(Some(E::ONE), |prev| Some(*prev * challenge))
        .take(columns.len() + 1)
        .collect::<Vec<E>>();
    // Compute the merged evaluations (to be used in calculating the multiplicity polynomial)
    (0..1 << num_vars)
        .into_iter()
        .map(|i| {
            column_evals
                .iter()
                .enumerate()
                .fold(challenge_powers[columns.len()], |acc, (j, col)| {
                    acc + challenge_powers[j] * col[i]
                })
        })
        .collect()
}
