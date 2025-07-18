//! File containing code for lookup witness generation.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    util::ceil_log2,
};
use p3_field::FieldAlgebra;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tracing::{debug, warn};
use transcript::Transcript;

use super::{logup_gkr::error::LogUpError, witness::LogUpWitness};
use crate::{
    Context, Element,
    iop::ChallengeStorage,
    layers::{
        activation::Relu,
        provable::{NodeId, ProvableOp},
    },
    model::{InferenceTrace, ToIterator},
    quantization::{self, Fieldizer},
};
use rayon::prelude::*;
pub const TABLE_POLY_ID_OFFSET: usize = 666;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// Enum used for establishing the different table types needed to prove non-linear functions in a model.
pub enum TableType {
    /// Table used for the Relu activation function
    Relu,
    /// Table used for range checking (its size is determined by the quantisation bit size)
    Range,
    /// Table used for clamping values, the inner [`usize`] denotes the maximum bit length a value can be before clamping to use this table
    Clamping(usize),
}

impl TableType {
    fn get_merged_table_column<E: ExtensionField>(
        &self,
        column_separator: Element,
    ) -> (Vec<Element>, Vec<Vec<E::BaseField>>) {
        match self {
            TableType::Relu => {
                #[allow(clippy::type_complexity)]
                let (comb, (col_one, col_two)): (
                    Vec<Element>,
                    (Vec<E::BaseField>, Vec<E::BaseField>),
                ) = (*quantization::MIN - 1..=*quantization::MAX)
                    .map(|i| {
                        let out = Relu::apply(i);
                        let i_field: E = i.to_field();
                        let out_field: E = out.to_field();
                        (
                            i + out * column_separator,
                            (i_field.as_bases()[0], out_field.as_bases()[0]),
                        )
                    })
                    .unzip();
                (comb, vec![col_one, col_two])
            }
            TableType::Range => {
                let (element_out, field): (Vec<Element>, Vec<E::BaseField>) = (0..1
                    << *quantization::BIT_LEN)
                    .map(|i| {
                        let i_field: E = i.to_field();
                        (i, i_field.as_bases()[0])
                    })
                    .unzip();
                (element_out, vec![field])
            }
            TableType::Clamping(size) => {
                let max = 1i128 << (size - 1);
                let min = -max;
                #[allow(clippy::type_complexity)]
                let (comb, (col_one, col_two)): (
                    Vec<Element>,
                    (Vec<E::BaseField>, Vec<E::BaseField>),
                ) = (min..max)
                    .map(|i| {
                        let out = if i < *quantization::MIN {
                            *quantization::MIN
                        } else if i > *quantization::MAX {
                            *quantization::MAX
                        } else {
                            i
                        };
                        let i_field: E = i.to_field();
                        let out_field: E = out.to_field();
                        (
                            i + out * column_separator,
                            (i_field.as_bases()[0], out_field.as_bases()[0]),
                        )
                    })
                    .unzip();
                (comb, vec![col_one, col_two])
            }
        }
    }

    pub fn name(&self) -> String {
        match self {
            TableType::Relu => "Relu".to_string(),
            TableType::Range => "Range".to_string(),
            TableType::Clamping(size) => format!("Clamping: {size}"),
        }
    }

    pub fn evaluate_table_columns<E: ExtensionField>(
        &self,
        point: &[E],
    ) -> Result<Vec<E>, LogUpError> {
        match self {
            TableType::Range => {
                if point.len() != *quantization::BIT_LEN {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a range table evaluation, point size: {}, expected: {}",
                        point.len(),
                        *quantization::BIT_LEN
                    )));
                }

                Ok(vec![
                    point.iter().enumerate().fold(E::ZERO, |acc, (index, p)| {
                        acc + *p * E::from_canonical_u64(1u64 << index)
                    }),
                ])
            }
            TableType::Relu => {
                if point.len() != *quantization::BIT_LEN {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a relu table evaluation, point size: {}, expected: {}",
                        point.len(),
                        *quantization::BIT_LEN
                    )));
                }

                let first_column = point.iter().enumerate().fold(E::ZERO, |acc, (index, p)| {
                    acc + *p * E::from_canonical_u64(1u64 << index)
                }) - E::from_canonical_u64(1u64 << (*quantization::BIT_LEN - 1));

                let second_column = point.iter().enumerate().take(point.len() - 1).fold(
                    E::ZERO,
                    |acc, (index, p)| {
                        acc + *p * E::from_canonical_u64(1u64 << index) * point[point.len() - 1]
                    },
                );
                Ok(vec![first_column, second_column])
            }
            TableType::Clamping(size) => {
                if point.len() != *size {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a clamping table evaluation, point size: {}, expected: {}",
                        point.len(),
                        size
                    )));
                }

                let first_column = point.iter().enumerate().fold(E::ZERO, |acc, (index, p)| {
                    acc + *p * E::from_canonical_u64(1u64 << index)
                }) - E::from_canonical_u64(1u64 << (size - 1));

                let max = 1i128 << (size - 1);
                let min = -max;

                let second_col_eval = (min..max)
                    .map(|i| {
                        let out = if i < *quantization::MIN {
                            *quantization::MIN
                        } else if i > *quantization::MAX {
                            *quantization::MAX
                        } else {
                            i
                        };

                        let out_field: E = out.to_field();
                        out_field.as_bases()[0]
                    })
                    .collect::<Vec<E::BaseField>>()
                    .into_mle()
                    .evaluate(point);

                Ok(vec![first_column, second_col_eval])
            }
        }
    }

    pub fn generate_challenge<E: ExtensionField, T: Transcript<E>>(&self, transcript: &mut T) -> E {
        match self {
            TableType::Relu => transcript.get_and_append_challenge(b"Relu").elements,
            TableType::Range => {
                // Theres only one column for a range check so we don't need to generate a challenge
                E::ONE
            }
            TableType::Clamping(_) => transcript.get_and_append_challenge(b"Clamping").elements,
        }
    }

    /// Gets the number of variables that the multiplicity polynomial will have for this table
    pub fn multiplicity_poly_vars(&self) -> usize {
        match self {
            TableType::Range | TableType::Relu => *quantization::BIT_LEN,
            TableType::Clamping(bits) => *bits,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupContext {
    tables: Vec<TableType>,
}

impl LookupContext {
    pub fn new(set: &BTreeSet<TableType>) -> LookupContext {
        LookupContext {
            tables: set.iter().copied().collect(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &TableType> {
        self.tables.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }
}

pub struct LookupWitnessGen<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    pub(crate) new_lookups: BTreeMap<TableType, Vec<Element>>,
    pub(crate) logup_witnesses: HashMap<NodeId, Vec<LogUpWitness<E, PCS>>>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> LookupWitnessGen<E, PCS> {
    pub fn new(lookup_ctx: &LookupContext) -> Self {
        let new_lookups = lookup_ctx
            .iter()
            .map(|&table_type| (table_type, Vec::<Element>::new()))
            .collect::<BTreeMap<TableType, Vec<Element>>>();
        Self {
            new_lookups,
            logup_witnesses: HashMap::new(),
        }
    }
}

pub(crate) const COLUMN_SEPARATOR: Element = 1i128 << 32;

pub struct LookupWitness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    pub challenge_storage: ChallengeStorage<E>,

    pub logup_witnesses: HashMap<NodeId, Vec<LogUpWitness<E, PCS>>>,

    pub table_witnesses: Vec<LogUpWitness<E, PCS>>,
}

pub fn generate_lookup_witnesses<'a, E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
    trace: &InferenceTrace<'a, E, Element>,
    ctx: &Context<E, PCS>,
    transcript: &mut T,
) -> Result<LookupWitness<E, PCS>, LogUpError>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    // If the lookup context is empty then there are no lookup witnesses to generate so we return default values
    if ctx.lookup.is_empty() {
        warn!("Lookup witness generation: no tables found, returning empty context TEST?");
        return Ok(LookupWitness {
            challenge_storage: ChallengeStorage {
                constant_challenge: E::ZERO,
                challenge_map: HashMap::new(),
            },
            logup_witnesses: HashMap::new(),
            table_witnesses: vec![],
        });
    }

    // Make the witness gen struct that stores relevant table lookup data
    let mut witness_gen = LookupWitnessGen::<E, PCS>::new(&ctx.lookup);

    debug!("Lookup witness generation: generating poly fields...");
    for (node_id, _) in ctx.steps_info.to_forward_iterator() {
        let step = trace
            .get_step(&node_id)
            .ok_or(LogUpError::ProvingError(format!(
                "Node {node_id} not found in trace"
            )))?;
        step.op
            .gen_lookup_witness(node_id, &mut witness_gen, ctx, &step.step_data)
            .map_err(|e| {
                LogUpError::ParameterError(format!(
                    "Error generating lookup witness for node {node_id} with error: {e}"
                ))
            })?;
    }

    debug!("Lookup witness generation: generating table multiplicities...");
    // calculate the table multiplicities
    let table_witnesses = witness_gen
        .new_lookups
        .par_iter()
        .map(|(table_type, lookups)| {
            let table_lookup_data =
                lookups
                    .iter()
                    .fold(HashMap::<Element, u64>::new(), |mut map, elem| {
                        *map.entry(*elem).or_insert(0) += 1;
                        map
                    });
            let (table_column, column_evals) =
                table_type.get_merged_table_column::<E>(COLUMN_SEPARATOR);

            let multiplicities = table_column
                .iter()
                .map(|table_val| {
                    if let Some(lookup_count) = table_lookup_data.get(table_val) {
                        E::BaseField::from_canonical_u64(*lookup_count)
                    } else {
                        E::BaseField::ZERO
                    }
                })
                .collect::<Vec<E::BaseField>>();
            let num_vars = ceil_log2(multiplicities.len());
            let mle =
                DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, &multiplicities);
            let commit = ctx.commitment_ctx.commit(&mle).map_err(|e| {
                LogUpError::PolynomialError(format!(
                    "Error while committing to {} table multiplicity polynomial: {:?}",
                    table_type.name(),
                    e
                ))
            })?;
            Ok(LogUpWitness::<E, PCS>::new_table(
                (commit, mle),
                multiplicities,
                column_evals,
                *table_type,
            ))
        })
        .collect::<Result<Vec<LogUpWitness<E, PCS>>, LogUpError>>()?;

    debug!("Lookup witness generation: commit context generation...");

    debug!("Lookup witness generation: challenge storage...");
    let challenge_storage =
        initialise_from_table_set::<E, T, _>(witness_gen.new_lookups.keys(), transcript);

    Ok(LookupWitness {
        challenge_storage,
        logup_witnesses: witness_gen.logup_witnesses,
        table_witnesses,
    })
}

fn initialise_from_table_set<
    'a,
    E: ExtensionField,
    T: Transcript<E>,
    I: Iterator<Item = &'a TableType>,
>(
    set: I,
    transcript: &mut T,
) -> ChallengeStorage<E> {
    let constant_challenge = transcript
        .get_and_append_challenge(b"table_constant")
        .elements;
    let challenge_map = set
        .map(|table_type| {
            let challenge = table_type.generate_challenge(transcript);

            (table_type.name(), challenge)
        })
        .collect::<HashMap<String, E>>();
    ChallengeStorage::<E> {
        constant_challenge,
        challenge_map,
    }
}
