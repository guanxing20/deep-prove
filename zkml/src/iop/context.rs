use crate::{
    iop::precommit::{self, PolyID}, layers::{provable::{ModelCtx, NodeCtx, ProvableModel}, LayerCtx}, lookup::context::{LookupContext, TableType}, Element
};
use anyhow::{anyhow, ensure, Context as CC};
use ff_ext::ExtensionField;
use mpcs::BasefoldCommitment;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{BTreeSet, HashMap};
use tracing::{debug, trace};
use transcript::Transcript;

/// Info related to the lookup protocol tables.
/// Here `poly_id` is the multiplicity poly for this table.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct TableCtx<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub poly_id: PolyID,
    pub num_vars: usize,
    pub table_commitment: BasefoldCommitment<E>,
}

/// Common information between prover and verifier
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Context<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub steps_info: ModelCtx<E>,
    /// Context related to the commitment and accumulation of claims related to the weights of model.
    /// This part contains the commitment of the weights.
    pub weights: precommit::Context<E>,
    /// Context holding all the different table types we use in lookups
    pub lookup: LookupContext,
}

/// Auxiliary information for the context creation
#[derive(Clone, Debug)]
pub(crate) struct ContextAux {
    pub tables: BTreeSet<TableType>,
    pub last_output_shape: Vec<Vec<usize>>,
}

impl<E: ExtensionField> Context<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Generates a context to give to the verifier that contains informations about the polynomials
    /// to prove at each step.
    /// INFO: it _assumes_ the model is already well padded to power of twos.
    /*pub fn generate(
        model: &Model<Element>,
        input_shape: Option<Vec<usize>>,
    ) -> anyhow::Result<Self> {
        let tables = BTreeSet::new();
        let last_output_shape = if let Some(shape) = input_shape {
            shape
        } else {
            model.input_shape()
        };
        let mut ctx_aux = ContextAux {
            tables,
            last_output_shape: vec![last_output_shape],
        };
        let mut step_infos = Vec::with_capacity(model.layer_count());
        debug!("Context : layer info generation ...");
        for (id, layer) in model.provable_layers() {
            trace!(
                "Context : {}-th layer {}info generation ...",
                id,
                layer.describe()
            );
            let (info, new_aux) = layer.step_info(id, ctx_aux);
            step_infos.push(info);
            ctx_aux = new_aux;
        }
        debug!("Context : commitment generating ...");
        let commit_ctx = precommit::Context::generate_from_model(model)
            .context("can't generate context for commitment part")?;
        debug!("Context : lookup generation ...");
        let lookup_ctx = LookupContext::new(&ctx_aux.tables);
        Ok(Self {
            steps_info: step_infos.into_iter().rev().collect_vec(),
            weights: commit_ctx,
            lookup: lookup_ctx,
        })
    }*/

    pub fn generate<T>(
        model: &ProvableModel<E, T, Element>,
        input_shapes: Option<Vec<Vec<usize>>>,
    ) -> anyhow::Result<Self> 
    where 
        T: Transcript<E>,
    {
        let tables = BTreeSet::new();
        let input_shapes = if let Some(shape) = input_shapes {
            shape
        } else {
            model.input_shapes.clone()
        };
        let mut ctx_aux = ContextAux {
            tables,
            last_output_shape: input_shapes.clone(),
        };
        let mut step_infos = HashMap::new();
        let mut shapes: HashMap<u64, Vec<Vec<usize>>> = HashMap::new();
        debug!("Context : layer info generation ...");
        for (id, node) in model.to_forward_iterator() {
            trace!(
                "Context : {}-th layer {}info generation ...",
                id,
                node.describe()
            );
            // compute input shapes for this node
            let node_input_shapes = node.inputs.iter().map(|edge| {
                Ok(if let Some(node_id) = &edge.node {
                    let node_shapes = shapes.get(node_id).ok_or(anyhow!("Node {} not found in set of previous shapes", node_id))?;
                    ensure!(edge.index < node_shapes.len(), "Input for node {} is coming from output {} of node {}, 
                        but this node has only {} outputs", id, edge.index, node_id, node_shapes.len());
                    node_shapes[edge.index].clone()
                } else {
                    // input node
                    ensure!(edge.index < input_shapes.len(), "Input for node {} is the input {} of the model, 
                        but the model has only {} inputs", id, edge.index, input_shapes.len());
                    input_shapes[edge.index].clone()
                })
            }).collect::<anyhow::Result<Vec<_>>>()?;
            ctx_aux.last_output_shape = node_input_shapes;
            let (info, new_aux) = node.step_info(id as PolyID, ctx_aux);
            step_infos.insert(id, NodeCtx {
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
                ctx: info,
            });
            ctx_aux =  new_aux;
            shapes.insert(id, ctx_aux.last_output_shape.clone());
        }

        debug!("Context : commitment generating ...");
        let commit_ctx = precommit::Context::generate_from_model(model)
            .context("can't generate context for commitment part")?;
        debug!("Context : lookup generation ...");
        let lookup_ctx = LookupContext::new(&ctx_aux.tables);
        Ok(Self {
            steps_info: ModelCtx { nodes: step_infos },
            weights: commit_ctx,
            lookup: lookup_ctx,
        })

    }

    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        for (_ , step_ctx) in self.steps_info.to_backward_iterator() {
            match &step_ctx.ctx {
                LayerCtx::Dense(info) => {
                    t.append_field_element(&E::BaseField::from(info.matrix_poly_id as u64));
                    info.matrix_poly_aux.write_to_transcript(t);
                }
                LayerCtx::Requant(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                }
                LayerCtx::Activation(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                }
                LayerCtx::Pooling(info) => {
                    t.append_field_element(&E::BaseField::from(info.poolinfo.kernel_size as u64));
                    t.append_field_element(&E::BaseField::from(info.poolinfo.stride as u64));
                }
                LayerCtx::Table(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                    t.append_field_elements(info.table_commitment.root().0.as_slice());
                }
                LayerCtx::Convolution(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.bias_poly_id as u64));

                    for i in 0..info.delegation_fft.len() {
                        info.delegation_fft[i].write_to_transcript(t);
                    }
                    for i in 0..info.delegation_ifft.len() {
                        info.delegation_ifft[i].write_to_transcript(t);
                    }
                    info.fft_aux.write_to_transcript(t);
                    info.ifft_aux.write_to_transcript(t);
                    info.hadamard.write_to_transcript(t);
                }
                LayerCtx::SchoolBookConvolution(_info) => {},
                LayerCtx::Reshape(_) => {},
            }
        }
        self.weights.write_to_transcript(t)?;
        Ok(())
    }
}
