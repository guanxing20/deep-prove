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
        cb.combine_columns(lookup_cell, &lookup_row, 1, 0);
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
