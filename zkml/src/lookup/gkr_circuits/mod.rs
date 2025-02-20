use ff::Field;
use ff_ext::ExtensionField;
use simple_frontend::structs::{CircuitBuilder, ExtCellId};

pub mod logup;
pub mod lookups_circuit;
pub mod table_circuit;

/// GKR circuit utility function to pairwise add a series of fractions of the form (p1, q1), (p2, q2) to obtain (p1q2+p2q1, q1q2)
pub(crate) fn add_fractions<E: ExtensionField>(
    cb: &mut CircuitBuilder<E>,
    numerators: &[ExtCellId<E>],
    denominators: &[ExtCellId<E>],
) -> (Vec<ExtCellId<E>>, Vec<ExtCellId<E>>) {
    numerators
        .chunks(2)
        .zip(denominators.chunks(2))
        .map(|(nums, denoms)| {
            let num_out = cb.create_ext_cell();
            cb.mul2_ext(&num_out, &nums[0], &denoms[1], E::BaseField::ONE);

            cb.mul2_ext(&num_out, &nums[1], &denoms[0], E::BaseField::ONE);

            let denom_out = cb.create_ext_cell();
            cb.mul2_ext(&denom_out, &denoms[0], &denoms[1], E::BaseField::ONE);
            (num_out, denom_out)
        })
        .unzip()
}
