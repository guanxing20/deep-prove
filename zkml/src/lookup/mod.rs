use std::collections::{BTreeMap, HashMap, HashSet};

use ff_ext::ExtensionField;
use gkr::structs::{Circuit, IOPProof};

use crate::{
    Claim,
    activation::Activation,
    iop::context::StepInfo,
    model::{InferenceStep, InferenceTrace, Layer, StepIdx},
};
use gkr_circuits::{
    logup::logup_circuit, lookups_circuit::lookup_wire_fractional_sumcheck,
    table_circuit::table_fractional_sumcheck,
};
use multilinear_extensions::mle::DenseMultilinearExtension;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

pub mod gkr_circuits;
pub mod logup;
pub mod utils;

pub use logup::LogUp;

type MLE<E> = DenseMultilinearExtension<E>;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof<E: ExtensionField> {
    // one commitment per "column" in the lookups
    // lookups_C: Vec<(Commitment<E>,Opening,Claim)>,
    input_column_claims: Vec<Claim<E>>,
    output_column_claims: Vec<Claim<E>>,
    m_claim: Claim<E>,
    gkr_proof: IOPProof<E>,
    output_claim: E,
    // multi_C: Commitment<E>,
}
impl<E: ExtensionField> Proof<E> {
    /// Retireve the claims about the input columns
    pub fn input_column_claims(&self) -> &[Claim<E>] {
        &self.input_column_claims
    }

    /// Retrieve the output column claims
    pub fn output_column_claims(&self) -> &[Claim<E>] {
        &self.output_column_claims
    }

    /// Retrieve the multiplicity poly claim
    pub fn multiplicity_claim(&self) -> &Claim<E> {
        &self.m_claim
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
            StepInfo::Requant(..) => LookupType::Range(8),
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
}

pub struct Context<E: ExtensionField> {
    circuits: HashMap<LookupType, Circuit<E>>,
    witness_ctx: WitnessContext<E>,
}

#[derive(Clone, Debug, Default)]
pub struct WitnessContext<E: ExtensionField> {
    current_step: StepIdx,
    witness_storage: BTreeMap<StepIdx, Vec<Vec<MLE<E>>>>,
    challenge_storage: HashMap<LookupType, (E, E)>,
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
}

impl<E: ExtensionField> WitnessContext<E> {
    /// Initialises from an [`InferenceTrace`]
    pub fn initialise(&mut self, trace: &InferenceTrace<E>) {
        trace.iter().for_each(|(step_input, step)| {
            let step_idx = step.id;
            match step.layer {
                Layer::Dense(..) => {}
                Layer::Activation(Activation::Relu(..)) => {}
                _ => unreachable!(),
            }
        })
    }
}

pub trait LookupProtocol<E: ExtensionField> {
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
