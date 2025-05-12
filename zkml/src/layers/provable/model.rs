use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tracing::info;
use transcript::Transcript;

use crate::{
    Element, IO, Tensor,
    layers::{
        Layer, LayerCtx,
        provable::{Edge, OpInfo},
        requant::Requant,
    },
    padding::PaddingMode,
    quantization::{Fieldizer, InferenceTracker, ModelMetadata, TensorFielder},
    tensor::Number,
    try_unzip,
};

use super::{Evaluate, InferenceStep, NodeCtx, NodeEgdes, NodeId, Op, ProvableNode, StepData};

#[derive(Debug, Clone)]
pub struct ProvableModel<N> {
    pub(crate) nodes: HashMap<NodeId, ProvableNode<N>>,
    pub(crate) input_shapes: Vec<Vec<usize>>,
    pub(crate) unpadded_input_shapes: Vec<Vec<usize>>,
}

pub struct Trace<'a, E: ExtensionField, N, D> {
    pub(crate) steps: HashMap<NodeId, InferenceStep<'a, E, N, D>>,
    pub(crate) input: Vec<Tensor<D>>,
    pub(crate) output: Vec<Tensor<D>>,
}
// The trace produce by running the model during inference
pub type InferenceTrace<'a, E: ExtensionField, N> = Trace<'a, E, N, N>;
// The trace used to prove the model
pub type ProvingTrace<'a, E: ExtensionField, N> = Trace<'a, E, N, E>;

impl<N> ProvableModel<N>
where
    N: Number,
{
    /// Returns an iterator over the nodes in the model, in arbitrary order.
    /// It is more efficient then `ForwardIterator` and `BackwardIterator`, so it
    /// can be used to iterate over the nodes when the order does not matter
    pub fn to_unstable_iterator(&self) -> impl Iterator<Item = (&NodeId, &ProvableNode<N>)> {
        self.nodes.iter()
    }

    fn compute_padded_input_shapes(unpadded_input_shapes: &[Vec<usize>]) -> Vec<Vec<usize>> {
        unpadded_input_shapes
            .into_iter()
            .map(|shape| {
                shape
                    .into_iter()
                    .map(|dim| dim.next_power_of_two())
                    .collect()
            })
            .collect()
    }

    pub(crate) fn new_from_input_shapes(
        unpadded_input_shapes: Vec<Vec<usize>>,
        padding: PaddingMode,
    ) -> Self {
        let input_shapes = match padding {
            PaddingMode::NoPadding => unpadded_input_shapes.clone(),
            PaddingMode::Padding => Self::compute_padded_input_shapes(&unpadded_input_shapes),
        };
        Self {
            nodes: HashMap::new(),
            input_shapes,
            unpadded_input_shapes,
        }
    }

    pub(crate) fn new<I: Iterator<Item = (NodeId, ProvableNode<N>)>>(
        unpadded_input_shapes: Vec<Vec<usize>>,
        padding: PaddingMode,
        nodes: I,
    ) -> Self {
        let mut model = Self::new_from_input_shapes(unpadded_input_shapes, padding);
        model.nodes = nodes.collect();

        model
    }

    pub(crate) fn new_from_shapes<I: Iterator<Item = (NodeId, ProvableNode<N>)>>(
        unpadded_input_shapes: Vec<Vec<usize>>,
        actual_input_shapes: Vec<Vec<usize>>,
        nodes: I,
    ) -> Self {
        Self {
            unpadded_input_shapes,
            input_shapes: actual_input_shapes,
            nodes: nodes.collect(),
        }
    }

    pub(crate) fn unpadded_input_shapes(&self) -> Vec<Vec<usize>> {
        self.unpadded_input_shapes.clone()
    }

    /// Get the actual input shapes
    pub fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.input_shapes.clone()
    }

    pub fn prepare_inputs(&self, inputs: Vec<Tensor<N>>) -> Result<Vec<Tensor<N>>> {
        let input_shapes = self.input_shapes.clone();
        ensure!(
            input_shapes.len() == inputs.len(),
            "Unexpected number of inputs tensors: expected {}, found {}",
            input_shapes.len(),
            inputs.len()
        );
        Ok(inputs
            .into_iter()
            .zip(input_shapes)
            .map(|(mut input, shape)| {
                if input.get_shape().clone() == shape {
                    // no need to pad, simply return the input
                    input
                } else {
                    input.pad_to_shape(shape);
                    input
                }
            })
            .collect())
    }

    pub fn load_input_flat(&self, input: Vec<Vec<N>>) -> Result<Vec<Tensor<N>>> {
        let input_tensor = input
            .into_iter()
            .zip(self.unpadded_input_shapes())
            .map(|(inp, shape)| Tensor::new(shape, inp))
            .collect();
        self.prepare_inputs(input_tensor)
    }

    pub(crate) fn padded_input_shapes(&self) -> Vec<Vec<usize>> {
        Self::compute_padded_input_shapes(&self.unpadded_input_shapes)
    }

    pub fn describe(&self) {
        info!("Model description:");
        for (idx, layer) in self.to_forward_iterator() {
            info!("\t - {}: {:?}", idx, layer.inputs);
            info!("\t- {}: {}", idx, layer.operation.describe());
        }
    }

    pub(crate) fn add_requant_node(
        &mut self,
        requant: Requant,
        input_node_id: NodeId,
    ) -> anyhow::Result<NodeId> {
        let input_node = self
            .nodes
            .get_mut(&input_node_id)
            .ok_or(anyhow!("Node {input_node_id} not found in the model"))?;
        let num_outputs = input_node.outputs.len();
        let requant_node = ProvableNode::new_with_outputs(
            (0..num_outputs)
                .map(|i| Edge {
                    node: Some(input_node_id),
                    index: i,
                })
                .collect(),
            Layer::Requant(requant),
            input_node.outputs.clone(), // copy output wires of `input_node` to requant node
        );
        // remove edges from outputs of `input_node`
        input_node.outputs = vec![Default::default(); num_outputs];
        let requant_id = self.add_node(requant_node)?;
        // route inputs of the nodes using outputs of `input_node_id` to the newly inserted
        // requant node
        let requant_node = self.nodes.get(&requant_id).ok_or(anyhow!(
            "Requant node {requant_id} just inserted not found in the model"
        ))?;
        for (i, wire) in requant_node.outputs.clone().iter().enumerate() {
            // change inputs of each node using this output wire
            wire.edges.iter().filter(|edge| edge.node.is_some()).try_for_each(|edge|{
                let node_id = edge.node.unwrap();
                let node = self.nodes.get_mut(&node_id).ok_or(
                    anyhow!("Node {node_id}, which should use an output of requant node {requant_id}, not found in model")
                )?;
                ensure!(edge.index < node.inputs.len(),
                    "Node {node_id} has {} inputs, so cannot access input {}",
                    node.inputs.len(),
                    edge.index,
                );
                // check that this input was indeed referring to an output of input_node_id
                let input_edge = &mut node.inputs[edge.index];
                ensure!(input_edge.node.ok_or(
                    anyhow!("{} input of node {node_id} should not be an input of the model", edge.index)
                )? == input_node_id,
                    "{} input of node {node_id} should be {input_node_id}", edge.index
                );
                // replace `input_node_id` with `requant_id`
                input_edge.node = Some(requant_id);
                input_edge.index = i;
                Ok(())
            })?;
        }
        Ok(requant_id)
    }

    /// Corner-case method to add a node whose inputs correspond to the outputs of a node already inserted in the model
    /// The `NodeId` of the already inserted node is the `previous_node_id` input; if no id is provided, it is assumed
    /// that the inputs of the node correspond to the inputs of the model
    pub(crate) fn add_consecutive_layer(
        &mut self,
        layer: Layer<N>,
        previous_node_id: Option<NodeId>,
    ) -> anyhow::Result<NodeId> {
        let num_outputs = if let Some(id) = &previous_node_id {
            let previous_node = self
                .nodes
                .get(id)
                .ok_or(anyhow!("Node {id} not found in model"))?;
            previous_node.outputs.len()
        } else {
            // correspond to inputs of the model
            self.input_shapes.len()
        };

        let new_node = ProvableNode::new(
            (0..num_outputs)
                .map(|i| Edge {
                    node: previous_node_id,
                    index: i,
                })
                .collect(),
            layer,
        );
        self.add_node(new_node)
    }

    pub(crate) fn add_node(&mut self, node: ProvableNode<N>) -> anyhow::Result<NodeId> {
        let node_id = self.nodes.len() as NodeId;
        // iterate over the inputs of the node and add the edges to the outputs of
        // corresponding nodes already in the model
        for (i, input) in node.inputs.iter().enumerate() {
            if let Some(input_node_id) = &input.node {
                let input_node = self.nodes.get_mut(input_node_id).ok_or(anyhow!(
                    "Node {} for input {} of new node not found in model",
                    input_node_id,
                    i,
                ))?;
                ensure!(
                    input.index < input_node.outputs.len(),
                    "Specified output number {} for node {}, which has only {} outputs",
                    input.index,
                    input_node_id,
                    input_node.outputs.len(),
                );
                input_node.outputs[input.index].edges.push(Edge {
                    node: Some(node_id),
                    index: i,
                });
            }
        }

        self.nodes.insert(node_id, node);

        Ok(node_id)
    }

    // Label the edges provided as input as the output edges of the model. If no edge is provided,
    // then the method assumes there is a node without routed output edges, and the outputs of
    // this node will be labelled as the output edges of the model
    pub(crate) fn route_output(&mut self, output_edges: Option<Vec<Edge>>) -> Result<()> {
        if let Some(output_edges) = output_edges {
            for (out_index, edge) in output_edges.iter().enumerate() {
                let out_node_id = edge
                    .node
                    .ok_or(anyhow!("Provided output edge with no input node"))?;
                let out_node = self
                    .nodes
                    .get_mut(&out_node_id)
                    .ok_or(anyhow!("Node {out_node_id} not found"))?;
                ensure!(
                    edge.index < out_node.outputs.len(),
                    "Specified output {} for node {out_node_id}, but only {} outputs found",
                    edge.index,
                    out_node.outputs.len()
                );
                out_node.outputs[edge.index].edges.push(Edge {
                    node: None,
                    index: out_index,
                })
            }
        } else {
            // find the node with no output edges, which will be considered the output node
            let out_node = self.nodes.iter_mut().find(|(id, node)| 
                node.outputs.iter().all(|out| 
                    out.clone() == Default::default()        
                )
            );
            ensure!(out_node.is_some(), "No output node found for model");
            let node = out_node.unwrap().1;
            node.outputs.iter_mut().enumerate().for_each(|(i, out)| 
                out.edges = vec![Edge { 
                    node: None, 
                    index: i,
                }]
            );
        }
        

        Ok(())
    }

    pub(crate) fn output_node(&self) -> NodeId {
        self.nodes
            .iter()
            .find_map(|(id, node)| {
                if node
                    .outputs
                    .iter()
                    .all(|wire| !wire.edges.is_empty() && wire.edges.iter().all(|edge| edge.node.is_none()))
                {
                    Some(*id)
                } else {
                    None
                }
            })
            .expect("No output node found")
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn get_node(&self, id: &NodeId) -> Option<&ProvableNode<N>> {
        self.nodes.get(id)
    }
}
// Creates an iterator over the nodes in the model, starting from the inputs and
// yielding nodes in order according to whether their inputs all come from nodes
// already visited by the iterator. This is useful for traversing the model when
// evaluating it at interence time
pub type ModelForwardIterator<'a, N> = NodeIterator<'a, ProvableNode<N>, true>;
// Creates an iterator over the nodes in the model, starting from the outputs and
// yielding nodes in order according to whether their outputs all come from nodes
// already visited by the iterator. This is useful for traversing the model when
// proving
pub type ModelBackwardIterator<'a, N> = NodeIterator<'a, ProvableNode<N>, false>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ModelCtx<E: ExtensionField + DeserializeOwned>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub(crate) nodes: HashMap<NodeId, NodeCtx<E>>,
}

impl<E: ExtensionField + DeserializeOwned> ModelCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub(crate) fn contexts(&self) -> impl Iterator<Item = &LayerCtx<E>> {
        self.nodes.iter().map(|(_, nodes)| &nodes.ctx)
    }
}

impl<E: ExtensionField + DeserializeOwned> ToIterator<NodeCtx<E>> for ModelCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn to_forward_iterator<'a>(&'a self) -> ModelCtxForwardIterator<'a, E> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            nodes: &self.nodes,
        }
    }

    fn to_backward_iterator<'a>(&'a self) -> ModelCtxBackwardIterator<'a, E> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            nodes: &self.nodes,
        }
    }
}

impl<N> ToIterator<ProvableNode<N>> for ProvableModel<N> {
    fn to_forward_iterator<'a>(&'a self) -> ModelForwardIterator<'a, N> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            nodes: &self.nodes,
        }
    }

    fn to_backward_iterator<'a>(&'a self) -> ModelBackwardIterator<'a, N> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            nodes: &self.nodes,
        }
    }
}

pub trait NodeCollection<E: NodeEgdes> {
    fn nodes(self) -> HashMap<NodeId, E>;
}

impl<N> NodeCollection<ProvableNode<N>> for ProvableModel<N> {
    fn nodes(self) -> HashMap<NodeId, ProvableNode<N>> {
        self.nodes
    }
}

pub type ModelCtxForwardIterator<'a, E> = NodeIterator<'a, NodeCtx<E>, true>;
pub type ModelCtxBackwardIterator<'a, E> = NodeIterator<'a, NodeCtx<E>, false>;

pub struct NodeIterator<'a, E: NodeEgdes, const FORWARD: bool> {
    pub(crate) unvisited_nodes: HashSet<NodeId>,
    pub(crate) nodes: &'a HashMap<NodeId, E>,
}

pub trait ToIterator<E: NodeEgdes> {
    fn to_forward_iterator<'a>(&'a self) -> NodeIterator<'a, E, true>;

    fn to_backward_iterator<'a>(&'a self) -> NodeIterator<'a, E, false>;

    fn into_forward_iterator(self) -> IntoNodeIterator<E, true>
    where
        Self: Sized + NodeCollection<E>,
    {
        IntoNodeIterator::new(self)
    }

    fn into_backward_iterator(self) -> IntoNodeIterator<E, false>
    where
        Self: Sized + NodeCollection<E>,
    {
        IntoNodeIterator::new(self)
    }
}

pub struct IntoNodeIterator<E: NodeEgdes, const FORWARD: bool> {
    pub(crate) node_ids: Vec<NodeId>,
    pub(crate) nodes: HashMap<NodeId, E>,
}

impl<E: NodeEgdes, const FORWARD: bool> IntoNodeIterator<E, FORWARD> {
    fn new<I: ToIterator<E> + NodeCollection<E>>(iter: I) -> Self {
        let mut node_ids: Vec<_> = if FORWARD {
            iter.to_forward_iterator()
                .map(|(node_id, _)| node_id)
                .collect()
        } else {
            iter.to_backward_iterator()
                .map(|(node_id, _)| node_id)
                .collect()
        };
        node_ids.reverse(); // reverse since we will pop elements from the end in implementation
        // of Iterator
        Self {
            node_ids,
            nodes: iter.nodes(),
        }
    }
}

impl<'a, E: NodeEgdes, const FORWARD: bool> Iterator for NodeIterator<'a, E, FORWARD> {
    type Item = (NodeId, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.unvisited_nodes.iter().find_map(|node_id| {
            let node = self.nodes.get(node_id).unwrap(); // safe to unwrap since this should contain only nodes in the model
            let is_node_next = if FORWARD {
                node.inputs().iter().all(|edge| {
                    edge.node.is_none()
                        || !self.unvisited_nodes.contains(edge.node.as_ref().unwrap())
                })
            } else {
                node.outputs()
                    .iter()
                    .flat_map(|output| &output.edges)
                    .all(|edge| {
                        edge.node.is_none()
                            || !self.unvisited_nodes.contains(edge.node.as_ref().unwrap())
                    })
            };
            if is_node_next {
                Some((*node_id, node))
            } else {
                None
            }
        });
        if let Some((node_id, _)) = &node {
            // remove the node from the unvisited nodes set
            self.unvisited_nodes.remove(node_id);
        }
        node
    }
}

impl<E: NodeEgdes, const FORWARD: bool> Iterator for IntoNodeIterator<E, FORWARD> {
    type Item = (NodeId, E);

    fn next(&mut self) -> Option<Self::Item> {
        if self.node_ids.is_empty() {
            None
        } else {
            let node_id = self.node_ids.pop().unwrap();
            let node = self
                .nodes
                .remove(&node_id)
                .expect(format!("Node {node_id} not found").as_str());
            Some((node_id, node))
        }
    }
}

impl<'a, E: ExtensionField, N, D> Trace<'a, E, N, D> {
    pub(crate) fn get_step(&self, node_id: &NodeId) -> Option<&InferenceStep<'a, E, N, D>> {
        self.steps.get(node_id)
    }

    pub(crate) fn new_step(&mut self, node_id: NodeId, step: InferenceStep<'a, E, N, D>) {
        self.steps.insert(node_id, step);
    }

    pub(crate) fn steps(&self) -> impl Iterator<Item = (&NodeId, &InferenceStep<'a, E, N, D>)> {
        self.steps.iter()
    }

    pub(crate) fn to_verifier_io(&self) -> IO<E>
    where
        D: Fieldizer<E>,
    {
        let input = self.input.iter().map(|inp| inp.to_fields()).collect();
        let output = self.output.iter().map(|out| out.to_fields()).collect();
        IO::new(input, output)
    }

    pub(crate) fn to_field(self) -> ProvingTrace<'a, E, N>
    where
        D: Fieldizer<E>,
    {
        let input = self.input.into_iter().map(|inp| inp.to_fields()).collect();
        let output = self.output.into_iter().map(|out| out.to_fields()).collect();
        let field_steps = self
            .steps
            .into_iter()
            .map(|(id, step)| {
                (id, InferenceStep {
                    op: step.op,
                    step_data: StepData {
                        inputs: step
                            .step_data
                            .inputs
                            .into_iter()
                            .map(|inp| inp.to_fields())
                            .collect(),
                        outputs: super::LayerOut {
                            outputs: step
                                .step_data
                                .outputs
                                .outputs
                                .into_iter()
                                .map(|out| out.to_fields())
                                .collect(),
                            proving_data: step.step_data.outputs.proving_data,
                        },
                        unpadded_output_shapes: step.step_data.unpadded_output_shapes,
                    },
                })
            })
            .collect();
        Trace {
            steps: field_steps,
            input,
            output,
        }
    }

    pub(crate) fn output(&self) -> Result<&Tensor<D>> {
        // we assume for now there is only one output tensor
        ensure!(
            self.output.len() == 1,
            "Found more than 1 output tensor for the model"
        );
        Ok(&self.output[0])
    }
}

impl<'a, E: ExtensionField> InferenceTrace<'a, E, Element> {
    pub fn dequantized(&self, md: &ModelMetadata) -> Trace<'a, E, Element, f32> {
        let inputs = self
            .input
            .iter()
            .zip(&md.input)
            .map(|(input, s)| input.dequantize(s))
            .collect();
        let outputs = self
            .output
            .iter()
            .zip(&md.output)
            .map(|(out, s)| out.dequantize(s))
            .collect();
        let steps = self
            .steps
            .iter()
            .map(|(node_id, step)| {
                let input_scaling = md.layer_input_scaling_factor(*node_id);
                let inputs = step
                    .step_data
                    .inputs
                    .iter()
                    .zip(input_scaling)
                    .map(|(inp, s)| inp.dequantize(s))
                    .collect();
                let output_scaling = md.layer_output_scaling_factor(*node_id);
                let outputs = step
                    .step_data
                    .outputs
                    .outputs()
                    .into_iter()
                    .zip(output_scaling)
                    .map(|(out, s)| out.dequantize(s))
                    .collect();
                (*node_id, InferenceStep {
                    op: step.op,
                    step_data: StepData {
                        inputs,
                        outputs: super::LayerOut {
                            outputs,
                            proving_data: step.step_data.outputs.proving_data.clone(),
                        },
                        unpadded_output_shapes: step.step_data.unpadded_output_shapes.clone(),
                    },
                })
            })
            .collect();
        Trace {
            steps,
            input: inputs,
            output: outputs,
        }
    }
}

impl<N: Number> ProvableModel<N> {
    pub(crate) fn run_with_tracker<E: ExtensionField>(
        &self,
        input: &[Tensor<N>],
        mut tracker: Option<&mut InferenceTracker>,
    ) -> anyhow::Result<InferenceTrace<'_, E, N>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
        Layer<N>: Evaluate<N>,
    {
        let mut trace = Trace {
            steps: HashMap::new(),
            input: input.to_vec(),
            output: vec![],
        };
        let iter = self.to_forward_iterator();
        for (node_id, node) in iter {
            let (inputs, shapes): (Vec<_>, Vec<_>) = try_unzip(node.inputs.iter().map(|edge| {
                Ok(if let Some(n) = &edge.node {
                    let step = trace.get_step(n);
                    ensure!(step.is_some(), "Node {} not found in trace", n);
                    let step = step.unwrap();
                    let outputs = step.step_data.outputs.outputs();
                    ensure!(
                        edge.index < outputs.len(),
                        "Requested output {} for node {}, which has only {} outputs",
                        edge.index,
                        n,
                        outputs.len()
                    );
                    // get shape of this output
                    let out_shapes = &step.step_data.unpadded_output_shapes;
                    (outputs[edge.index], out_shapes[edge.index].clone())
                } else {
                    (
                        &input[edge.index],
                        self.unpadded_input_shapes()[edge.index].clone(),
                    )
                })
            }))?;
            let output_shapes = node
                .operation
                .output_shapes(&shapes, PaddingMode::NoPadding);
            let output = node.run(inputs.as_slice(), shapes)?;
            // add output tensors to tracker, if any
            if let Some(tracker) = &mut tracker {
                for (i, out) in output.outputs().into_iter().enumerate() {
                    tracker.track(node_id, i, out.to_f32()?);
                }
            }
            let new_step = StepData {
                inputs: inputs.into_iter().cloned().collect(),
                outputs: output,
                unpadded_output_shapes: output_shapes,
            };
            trace.new_step(node_id, InferenceStep {
                op: &node.operation,
                step_data: new_step,
            });
        }

        // compute the output tensor from the outputs of the output nodes
        let output_node = self.output_node();
        let output: Vec<Tensor<N>> = trace
            .get_step(&output_node)
            .ok_or(anyhow!("Output node {} not found in trace", output_node))?
            .outputs()
            .into_iter()
            .cloned()
            .collect();
        ensure!(output.len() == 1, "Unexpected number of output tensors");

        trace.output = output;

        Ok(trace)
    }

    pub(crate) fn run<E: ExtensionField>(
        &self,
        input: &[Tensor<N>],
    ) -> anyhow::Result<InferenceTrace<'_, E, N>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
        Layer<N>: Evaluate<N>,
    {
        self.run_with_tracker(input, None)
    }

    pub(crate) fn provable_nodes(&self) -> impl Iterator<Item = (&NodeId, &ProvableNode<N>)> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.operation.is_provable())
    }
}

#[cfg(test)]
mod tests {
    use goldilocks::GoldilocksExt2;
    use transcript::BasicTranscript;

    use crate::{
        Context, Element, IO, Prover, ScalingStrategy, Tensor, init_test_logging,
        layers::{
            Layer,
            activation::{Activation, Relu},
            dense::Dense,
            provable::{Edge, OpInfo, ProvableNode},
        },
        padding::{PaddingMode, pad_model},
        quantization::{InferenceObserver, TensorFielder},
        tensor::Number,
        testing::random_vector,
        verify,
    };

    use super::ProvableModel;

    type E = GoldilocksExt2;
    type T = BasicTranscript<GoldilocksExt2>;
    type N = Element;

    fn build_test_model<N: Number, const INPUT_SIZE: usize>() -> ProvableModel<N> {
        let input_shape = vec![INPUT_SIZE];
        let mut model = ProvableModel::<N>::new_from_input_shapes(
            vec![input_shape.clone()],
            PaddingMode::NoPadding,
        );
        // add input dense layer
        // generate random dense matrix
        let ncols = input_shape[0];
        let nrows = 42;
        let dense = Dense::random(vec![nrows, ncols]);
        let dense_out_shape =
            &dense.output_shapes(&model.unpadded_input_shapes(), PaddingMode::NoPadding)[0];
        let input_node = model
            .add_consecutive_layer(
                Layer::Dense(dense),
                None, // it's connected to the inputs of the model
            )
            .unwrap();
        // add activation layer
        let relu = Activation::Relu(Relu::new());
        let relu_node = model
            .add_consecutive_layer(Layer::Activation(relu), Some(input_node))
            .unwrap();
        // add another dense layer as output
        let nrows = 37;
        let ncols = dense_out_shape[0]; // it's a vector, so it has only one dimension
        let dense = Dense::random(vec![nrows, ncols]);
        let output_node = model
            .add_consecutive_layer(Layer::Dense(dense), Some(relu_node))
            .unwrap();
        model
            .route_output(None)
            .unwrap();

        assert_eq!(model.output_node(), output_node);

        model
    }

    #[test]
    fn test_model_inference() {
        const INPUT_SIZE: usize = 45;
        let model = build_test_model::<N, INPUT_SIZE>();
        let input_shape = model.input_shapes()[0].clone();

        let input = random_vector(input_shape.iter().product());
        let input_tensor = Tensor::new(input_shape, input);
        let trace = model.run::<E>(&[input_tensor]).unwrap();
        assert_eq!(trace.steps.len(), 3);
    }

    #[test]
    fn test_model_float_inference() {
        const INPUT_SIZE: usize = 45;
        let model = build_test_model::<f32, INPUT_SIZE>();
        let input_shape = model.input_shapes()[0].clone();

        let input_tensor = Tensor::random(&input_shape);
        let trace = model.run::<E>(&[input_tensor]).unwrap();
        assert_eq!(trace.steps.len(), 3);
    }

    #[test]
    fn test_model_proving() {
        init_test_logging();
        const INPUT_SIZE: usize = 57;
        let model = build_test_model::<f32, INPUT_SIZE>();
        let input_shape = model.input_shapes()[0].clone();

        let float_input_tensor = Tensor::random(&input_shape);
        let (quantized_model, md) = InferenceObserver::new().quantize(model).unwrap();
        let model = pad_model(quantized_model).unwrap();

        model.describe();

        // quantize and pad input tensor
        let input_tensor = float_input_tensor
            .quantize(&md.input[0])
            .pad_next_power_of_two();

        let trace = model.run(&[input_tensor]).unwrap();
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"model");
        let ctx =
            Context::<GoldilocksExt2>::generate(&model, None).expect("Unable to generate context");
        let prover: Prover<'_, E, T> = Prover::new(&ctx, &mut tr);
        let io = trace.to_verifier_io();
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"model");
        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }
}
