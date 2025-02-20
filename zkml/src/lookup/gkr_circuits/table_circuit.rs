//! Contains code for building a GKR to obtain the output of a fractional sumcheck for the final table and mutliplicity poly when using the LogUp protocol.

use ff::Field;
use ff_ext::ExtensionField;
use gkr::structs::Circuit;

use simple_frontend::structs::{CellId, CircuitBuilder, ExtCellId};

use super::add_fractions;

pub fn table_fractional_sumcheck<E: ExtensionField>(
    table_partitioning: &[usize],
    num_vars: usize,
) -> Circuit<E> {
    let cb = &mut CircuitBuilder::default();
    let no_table_columns = table_partitioning.iter().sum::<usize>();

    // For each column of the table we have an input witness
    let table_columns = (0..no_table_columns)
        .map(|_| cb.create_witness_in(1 << num_vars).1)
        .collect::<Vec<Vec<CellId>>>();

    let multiplicity_poly = cb.create_witness_in(1 << num_vars).1;

    let table = cb.create_ext_cells(1 << num_vars);

    let challenge_ids = (0..table_partitioning.len())
        .map(|j| j as u8)
        .collect::<Vec<u8>>();

    let const_challenge_id = table_partitioning.len() as u8;

    table.iter().enumerate().for_each(|(i, table_cell)| {
        let table_row = table_columns
            .iter()
            .map(|col| col[i])
            .collect::<Vec<CellId>>();
        cb.calculate_table_column(
            table_cell,
            &table_row,
            table_partitioning,
            &challenge_ids,
            const_challenge_id,
        );
    });

    let (mut table_nums, mut table_denoms): (Vec<ExtCellId<E>>, Vec<ExtCellId<E>>) =
        multiplicity_poly
            .chunks(2)
            .zip(table.chunks(2))
            .map(|(nums, denoms)| {
                let num = cb.create_ext_cell();
                cb.mul_ext_base(&num, &denoms[0], nums[1], E::BaseField::ONE);
                cb.mul_ext_base(&num, &denoms[1], nums[0], E::BaseField::ONE);

                let denom = cb.create_ext_cell();
                cb.mul2_ext(&denom, &denoms[0], &denoms[1], E::BaseField::ONE);
                (num, denom)
            })
            .unzip();

    while table_nums.len() > 1 && table_denoms.len() > 1 {
        (table_nums, table_denoms) = add_fractions(cb, &table_nums, &table_denoms);
    }

    table_nums
        .into_iter()
        .zip(table_denoms.into_iter())
        .for_each(|(num, denom)| {
            cb.create_witness_out_from_exts(&[num]);
            cb.create_witness_out_from_exts(&[denom]);
        });

    cb.configure();

    Circuit::new(cb)
}
