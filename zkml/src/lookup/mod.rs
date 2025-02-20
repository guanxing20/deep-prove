use anyhow::anyhow;
use ff::Field;
use ff_ext::ExtensionField;
use gkr::structs::{Circuit, CircuitWitness, IOPProof};
use mpcs::{
    BasefoldCommitment, BasefoldCommitmentWithWitness, PolynomialCommitmentScheme,
    util::num_of_bytes,
};
use poseidon::poseidon_hash::hash_n_to_hash_no_pad;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{
    Claim,
    activation::Activation,
    commit::Pcs,
    iop::context::StepInfo,
    model::{InferenceStep, InferenceTrace, Layer, StepIdx},
};
use gkr_circuits::{
    logup::logup_circuit, lookups_circuit::lookup_wire_fractional_sumcheck,
    table_circuit::table_fractional_sumcheck,
};
use multilinear_extensions::mle::{DenseMultilinearExtension, FieldType};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

pub mod gkr_circuits;
pub mod logup;
pub mod utils;

pub use logup::LogUp;

type MLE<E> = DenseMultilinearExtension<E>;

const NO_LOOKUP_DOM_SEP: &'static [u8] = b"";
const RANGE_CHECK_DOM_SEP: &'static [u8] = b"range_check";
const RELU_DOM_SEP: &'static [u8] = b"relu";
const FINAL_TABLE_DOM_SEP: &'static [u8] = b"";

/// Proof from a GKR based lookup.
/// The commitment is a batch commitment to all of the witness wires.
/// When this is a "normal" lookup proof (not the final table) this is a batch commitment to all of the inputs
/// if its the final table proof its just a commitment to the multiplicity poly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// one commitment for all columns in the lookups
    commitment: BasefoldCommitment<E>,
    /// Claims about the witness polynomials
    claims: Vec<Claim<E>>,
    /// The actual GKR proof
    gkr_proof: IOPProof<E>,
    /// numerators for all the fractional sumchecks, in the same order as the claims for witness polys
    numerators: Vec<E>,
    /// denominators for all the fractional sumchecks, in the same order as the claims for witness polys
    denominators: Vec<E>,
}
impl<E: ExtensionField> Proof<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// Retireve the claims about the input columns
    pub fn claims(&self) -> &[Claim<E>] {
        &self.claims
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierClaims<E: ExtensionField> {
    input_claims: Vec<Claim<E>>,
    output_claims: Vec<Claim<E>>,
    multiplicity_claim: Claim<E>,
}

impl<E: ExtensionField> VerifierClaims<E> {
    pub fn input_claims(&self) -> &[Claim<E>] {
        &self.input_claims
    }

    pub fn output_claims(&self) -> &[Claim<E>] {
        &self.output_claims
    }

    pub fn multiplicity_poly_claim(&self) -> &Claim<E> {
        &self.multiplicity_claim
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LookupType {
    Relu,
    Range(u8),
    FinalTable(Vec<usize>, usize),
    NoLookup,
}

impl<E: ExtensionField> From<&StepInfo<E>> for LookupType {
    fn from(value: &StepInfo<E>) -> LookupType {
        match value {
            StepInfo::Requant(info) => LookupType::Range(info.shift as u8),
            StepInfo::Activation(info) => LookupType::from(&info.op),
            _ => LookupType::NoLookup,
        }
    }
}

impl From<&Activation> for LookupType {
    fn from(_: &Activation) -> Self {
        LookupType::Relu
    }
}

impl LookupType {
    pub fn make_circuit<E: ExtensionField>(&self) -> Circuit<E> {
        match self {
            LookupType::Range(bit_size) => lookup_wire_fractional_sumcheck(1, *bit_size as usize),
            LookupType::Relu => lookup_wire_fractional_sumcheck(2, 8),
            LookupType::FinalTable(table_partitioning, table_vars) => {
                table_fractional_sumcheck(table_partitioning, *table_vars)
            }
            LookupType::NoLookup => Circuit::<E>::default(),
        }
    }

    pub fn number_of_columns(&self) -> usize {
        match self {
            LookupType::Range(..) => 1,
            LookupType::Relu => 2,
            LookupType::FinalTable(partition, ..) => partition.iter().sum::<usize>(),
            LookupType::NoLookup => 0,
        }
    }

    pub fn num_vars(&self) -> usize {
        match self {
            LookupType::Range(bit_size) => *bit_size as usize,
            LookupType::Relu => 8,
            LookupType::FinalTable(.., table_num_vars) => *table_num_vars,
            LookupType::NoLookup => 0,
        }
    }

    pub fn num_witness_mles(&self) -> usize {
        match self {
            LookupType::FinalTable(partition, ..) => partition.iter().sum::<usize>() + 1,
            _ => self.number_of_columns(),
        }
    }

    pub fn get_dom_sep(&self) -> &[u8] {
        match &self {
            LookupType::NoLookup => NO_LOOKUP_DOM_SEP,
            LookupType::Range(..) => RANGE_CHECK_DOM_SEP,
            LookupType::Relu => RELU_DOM_SEP,
            LookupType::FinalTable(..) => FINAL_TABLE_DOM_SEP,
        }
    }
}

pub struct Context<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    circuits: HashMap<LookupType, Circuit<E>>,
    witness_ctx: WitnessContext<E>,
}

#[derive(Clone, Debug, Default)]
pub struct WitnessContext<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    current_step: StepIdx,
    witness_storage: BTreeMap<StepIdx, LookupProverInfo<E>>,
}

pub struct LookupProverInfo<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    pub lookup_type: LookupType,
    pub batch_commitment: BasefoldCommitmentWithWitness<E>,
    pub challenge: E,
    pub const_challenge: E,
    pub circuit_witness: CircuitWitness<E>,
}

impl<E> Context<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Getter for the lookup circuit depending on the [`StepInfo`]
    pub fn get_circuit(&self, step_info: &StepInfo<E>) -> anyhow::Result<&Circuit<E>> {
        let lookup_type: LookupType = step_info.try_into()?;
        self.circuits.get(&lookup_type).ok_or(anyhow::anyhow!(
            "Context does not contain a circuit for the lookup type: {:?}",
            lookup_type
        ))
    }

    /// Generate [`Context`] from a [ModelContext](`crate::iop::context::Context`)
    pub fn generate(ctx: &crate::iop::context::Context<E>) -> Context<E> {
        let mut keys_hash_set =
            HashSet::<LookupType>::from_iter(ctx.steps_info.iter().map(LookupType::from));

        keys_hash_set.remove(&LookupType::NoLookup);

        let (table_partition, table_num_vars) =
            keys_hash_set
                .iter()
                .fold((vec![], 0), |(acc_partition, current_vars), lookup_type| {
                    let (lookup_cols, num_vars) =
                        (lookup_type.number_of_columns(), lookup_type.num_vars());
                    (
                        [acc_partition, vec![lookup_cols]].concat(),
                        std::cmp::max(num_vars, current_vars),
                    )
                });

        keys_hash_set.insert(LookupType::FinalTable(table_partition, table_num_vars));

        let circuits = keys_hash_set
            .into_iter()
            .map(|k| {
                let circuit = k.make_circuit();
                (k, circuit)
            })
            .collect::<HashMap<LookupType, Circuit<E>>>();

        Context {
            circuits,
            witness_ctx: WitnessContext::<E>::default(),
        }
    }

    /// Initialises [`WitnessContext`] from an [`InferenceTrace`]
    pub fn initialise_witness_ctx(&mut self, trace: &InferenceTrace<E>) -> anyhow::Result<()> {
        // First we quickly iterate through all the steps to find the largest number of variables that will be used as input to a lookup

        let max_vars = trace
            .iter()
            .fold(0, |current_max, (_, step)| match step.layer {
                Layer::Dense(..) => current_max,
                Layer::Activation(Activation::Relu(..)) => std::cmp::max(current_max, 8),
                Layer::Requant(info) => std::cmp::max(current_max, info.right_shift),
                _ => unreachable!(),
            });

        // make the basefold params to commit to all the polys (we won't open them this is just to save some hashing in the verifier).
        let params = Pcs::<E>::setup(1 << max_vars)?;

        // For each step in the inference trace construct the witness MLEs and also batch commit to them so we can generate seperation challenges.
        trace.iter().try_for_each(|(step_input, step)| {
            let step_idx = step.id;
            match step.layer {
                Layer::Dense(..) => Result::<(), anyhow::Error>::Ok(()),
                Layer::Activation(Activation::Relu(..)) => {
                    let mles = [step_input, step.output.as_slice()]
                        .iter()
                        .map(|val| {
                            DenseMultilinearExtension::from_evaluations_vec(
                                8,
                                val.iter()
                                    .map(|eval| eval.as_bases()[0])
                                    .collect::<Vec<E::BaseField>>(),
                            )
                        })
                        .collect::<Vec<MLE<E>>>();
                    let (pp, _) = Pcs::<E>::trim(params.clone(), 1 << 8)?;
                    let batch_commit = Pcs::<E>::batch_commit(&pp, &mles)?;
                    self.witness_ctx
                        .witness_storage
                        .insert(step_idx, (LookupType::Relu, batch_commit));
                    Ok(())
                }
                Layer::Requant(requant) => {
                    // We ceiling divide the right shift by 8 and add one for now (because the quant size is constant currently)
                    // TODO: update this to work if we let quant size vary. This gives us the number of chunks we split the part we discard into.
                    let num_discarded_chunks = (requant.right_shift as usize - 1) / 8 + 1;
                    // We have one more chunk which is the part that we will keep
                    let num_chunks = num_discarded_chunks + 1;
                    let padded_chunks = num_chunks.next_power_of_two();

                    let mut relu_input = vec![];
                    let mut discarded_chunks = vec![vec![]; num_discarded_chunks];

                    // Make a mask for so we can get ridd of the relu input
                    let top_mask = 255u64 << requant.right_shift;
                    // Bit mask for the bytes
                    let bit_mask = 255u64;
                    step_input.iter().for_each(|val| {
                        // First we take the relu input
                        let u64val = val.to_canonical_u64_vec()[0];
                        relu_input.push(E::BaseField::from(u64val >> requant.right_shift));
                        // the value of an input should always be basefield elements
                        let mut masked_val = u64val & top_mask;
                        (0..num_discarded_chunks).rev().for_each(|j| {
                            let chunk = masked_val & bit_mask;
                            discarded_chunks[j].push(E::BaseField::from(chunk));
                            masked_val >>= 8;
                        })
                    });

                    discarded_chunks.insert(0, relu_input);

                    let num_vars = step_input.len().next_power_of_two().ilog2() as usize;
                    let witness_evals = [
                        vec![vec![E::BaseField::ZERO; 1 << num_vars]; padded_chunks - num_chunks],
                        discarded_chunks,
                    ]
                    .concat();
                    let (pp, _) = Pcs::<E>::trim(params.clone(), 1 << num_vars)?;
                    let witness_mles = witness_evals
                        .iter()
                        .map(|evals| {
                            DenseMultilinearExtension::from_evaluations_slice(num_vars, evals)
                        })
                        .collect::<Vec<MLE<E>>>();
                    let batch_commit = Pcs::<E>::batch_commit(&pp, &witness_mles)?;
                    self.witness_ctx.witness_storage.insert(
                        step_idx,
                        (LookupType::Range(requant.right_shift as u8), batch_commit),
                    );
                    Ok(())
                }
                _ => unreachable!(),
            }
        })?;

        // Now we work out the final challenges using the commitments
        let commits_in_order = self
            .witness_ctx
            .witness_storage
            .iter()
            .flat_map(|(_, (_, comm))| comm.get_root_as().0)
            .collect::<Vec<E::BaseField>>();

        let constant_challenge = E::from(hash_n_to_hash_no_pad(&commits_in_order.clone()).0[0]);

        let challenge_hash_map =
            HashMap::<LookupType, (E, E)>::from_iter(self.circuits.keys().map(|lt| {
                let extra = lt
                    .get_dom_sep()
                    .iter()
                    .map(|byte| E::BaseField::from(*byte as u64))
                    .collect::<Vec<E::BaseField>>();
                let input = [commits_in_order.clone(), extra].concat();
                let challenge = E::from(hash_n_to_hash_no_pad(&input).0[0]);
                (lt.clone(), (challenge, constant_challenge))
            }));
        // Store them all in the witness context
        self.witness_ctx.challenge_storage = challenge_hash_map;

        // Now we work out the witness for the final table GKR circuit
        // let all_lookups = self.witness_ctx.witness_storage.values().flat_map(|(lt, comm)| {
        //     let (challenge, const_challenge) = self.witness_ctx.challenge_storage.get(lt).ok_or(anyhow!("No challenges stored for lookup type: {:?}", lt))?;
        //     let evals = comm.get_evals_ref().iter().map(|ft| match ft {
        //         FieldType::Base(inner) => inner.clone(),
        //         _ => unreachable!(),
        //     }).collect::<Vec<Vec<E::BaseField>>>();
        //     let size = comm.poly_size();
        //     (0..size).map(|j| evals.iter().enumerate())
        // })
        todo!()
    }
}

pub trait LookupProtocol<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    // lookups has dimension (N,N) in case of two columns
    // with N lookup "op".
    // we must pad lookups MLEs to same dim than table
    // table can come from context
    // e.g table[i].num_vars() == lookups[i].num_vars()
    fn prove<T: Transcript<E>>(
        lookup_ctx: &Context<E>,
        lookup_type: &LookupType,
        mles: &[MLE<E>],
        t: &mut T,
    ) -> anyhow::Result<Proof<E>> {
        todo!()
    }

    // commitments to the lookups, one commitment per "column"
    fn verify<T: Transcript<E>>(
        lookup_ctx: &Context<E>,
        lookup_type: &LookupType,
        proof: Proof<E>,
        t: &mut T,
    ) -> anyhow::Result<VerifierClaims<E>> {
        todo!()
    }
}
