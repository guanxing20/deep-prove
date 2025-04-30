use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, ensure, Result};
use ff_ext::ExtensionField;
use serde::{de::DeserializeOwned, Serialize};
use transcript::Transcript;

use crate::{layers::provable::Edge, quantization::{Fieldizer, TensorFielder}, tensor::Number, Tensor};

use super::{InferenceStep, Node, NodeId, ProvableNode, ProvableOp};

pub struct Model<Op> {
    pub(crate) nodes: HashMap<NodeId, Node<Op>>,
    pub(crate) input_shapes: Vec<Vec<usize>>,
}

pub type ProvableModel<E, T, D> = Model<Box<dyn ProvableOp<E, T, D>>>;

/*pub struct ProvableModel<E, T, D> {
    pub(crate) nodes: HashMap<NodeId, ProvableNode<E, T, D>>,
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_shape: Vec<usize>,
}*/

pub struct InferenceTrace<N, E: ExtensionField> {
    pub(crate) steps: HashMap<NodeId, InferenceStep<N, E>>,
    pub(crate) input: Vec<Tensor<N>>,
    pub(crate) output: Vec<Tensor<N>>,
}

impl<Op> Model<Op> {
    pub fn to_forward_iterator(&self) -> ForwardIterator<Op> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            model: self,
        }
    }

    pub fn to_backward_iterator(&self) -> BackwardIterator<Op> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            model: self,
        }
    }

    /// Returns an iterator over the nodes in the model, in arbitrary order. 
    /// It is more efficient then `ForwardIterator` and `BackwardIterator`, so it
    /// can be used to iterate over the nodes when the order does not matter
    pub fn to_unstable_iterator(&self) -> impl Iterator<Item = (&NodeId, &Node<Op>)> {
        self.nodes.iter()
    }

    pub(crate) fn new_from_input_shapes(
        input_shapes: Vec<Vec<usize>>
    ) -> Self {
        Self {
            nodes: HashMap::new(),
            input_shapes,
        }
    }

    pub(crate) fn add_node(
        &mut self,
        node: Node<Op>,
    ) -> anyhow::Result<NodeId> {
        let node_id = self.nodes.len() as NodeId;
        // iterate over the inputs of the node and add the edges to the outputs of
        // corresponding nodes already in the model
        for (i, input) in node.inputs.iter().enumerate() {
            if let Some(node_id) = &input.node {
                let input_node = self.nodes.get_mut(node_id).ok_or(
                    anyhow!("Node {} for input {} of new node not found in model",
                        node_id,
                    i,
                ))?;
                ensure!(input.index < input_node.outputs.len(),
                    "Specified output number {} for node {}, which has only {} outputs",
                    input.index, node_id, input_node.outputs.len(),
                );
                input_node.outputs[input.index].edges.push(
                    Edge {
                        node: Some(*node_id),
                        index: i,
                    }
                );
            }
        }
        
        self.nodes.insert(node_id, node);
        
        Ok(node_id)
    }

    pub(crate) fn output_node(&self) -> NodeId {
        self.nodes.iter().find_map(|(id, node)|
            if node.outputs.iter().all(|wire|
                wire.edges.iter().all(|edge| edge.node.is_none())
            ) {
                Some(*id)
            } else {
                None
            }
        ).expect("No output node found")
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.nodes.len()
    }
} 
// Creates an iterator over the nodes in the model, starting from the inputs and
// yielding nodes in order according to whether their inputs all come from nodes
// already visited by the iterator. This is useful for traversing the model when
// evaluating it at interence time
pub type ForwardIterator<'a, Op> = NodeIterator<'a, Op, true>;
// Creates an iterator over the nodes in the model, starting from the outputs and
// yielding nodes in order according to whether their outputs all come from nodes
// already visited by the iterator. This is useful for traversing the model when
// proving
pub type BackwardIterator<'a, Op> = NodeIterator<'a, Op, false>;


pub struct NodeIterator<'a, Op, const FORWARD: bool> {
    pub(crate) unvisited_nodes: HashSet<NodeId>,
    pub(crate) model: &'a Model<Op>,
}

impl<'a, Op, const FORWARD: bool> Iterator for NodeIterator<'a, Op, FORWARD> {
    type Item = (NodeId, &'a Node<Op>);

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.unvisited_nodes.iter().find_map(|node_id| {
            let node = self.model.nodes.get(node_id)
                .unwrap(); // safe to unwrap since this should contain only nodes in the model
            let is_node_next = if FORWARD {
                node.inputs.iter().all(|edge| 
                    edge.node.is_none() || !self.unvisited_nodes.contains(edge.node.as_ref().unwrap())
                )
            } else {
                node.outputs.iter().flat_map(|output| &output.edges).all(|edge| 
                    edge.node.is_none() || !self.unvisited_nodes.contains(edge.node.as_ref().unwrap())
                )
            };
            if is_node_next {
                Some((*node_id, node))
            } else {
                None
            }
        } 
        );
        if let Some((node_id, _)) = &node {
            // remove the node from the unvisited nodes set
            self.unvisited_nodes.remove(node_id);
        }
        node
    }
}

impl<N, E: ExtensionField> InferenceTrace<N, E> {
    pub(crate) fn get_step(&self, node_id: &NodeId) -> Option<&InferenceStep<N, E>> {
        self.steps.get(node_id)
    }

    pub(crate) fn new_step(&mut self, node_id: NodeId, step: InferenceStep<N, E>) {
        self.steps.insert(node_id, step);
    }

    pub(crate) fn to_field(self) -> InferenceTrace<E, E> 
    where N: Fieldizer<E>,
    {
        let input = self.input.into_iter().map(|inp| inp.to_fields()).collect();
        let output = self.output.into_iter().map(|out| out.to_fields()).collect();
        let field_steps = self
            .steps
            .into_iter()
            .map(|(id, step)|
                (
                    id, 
                    InferenceStep {
                        inputs: step.inputs.into_iter().map(|inp| inp.to_fields()).collect(),
                        outputs: super::LayerOut { 
                            outputs: step.outputs.outputs.into_iter().map(|out| out.to_fields()).collect(), 
                            proving_data: step.outputs.proving_data, 
                        }
                    }
                )
            ).collect();
        InferenceTrace {
            steps: field_steps,
            input,
            output,
        }
    }

    pub(crate) fn output(&self) -> Result<&Tensor<N>> {
        // we assume for now there is only one output tensor
        ensure!(self.output.len() == 1, "Found more than 1 output tensor for the model");
        Ok(&self.output[0])
    }
}

impl<E, T, N> ProvableModel<E, T, N> 
where 
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    T: Transcript<E>,
    N: Number,
{
    pub(crate) fn run(&self, input: &[Tensor<N>]) -> anyhow::Result<InferenceTrace<N, E>> {
        let mut trace = InferenceTrace {
            steps: HashMap::new(),
            input: input.to_vec(),
            output: vec![],
        };
        let iter = self.to_forward_iterator();
        for (node_id, node) in iter {
            let inputs = node.inputs.iter().map(|edge|
                Ok(if let Some(n) = &edge.node {
                    let step = trace.get_step(n);
                    ensure!(step.is_some(), "Node {} not found in trace", n);
                    let step = step.unwrap();
                    let outputs = step.outputs.outputs();
                    outputs[edge.index]
                } else {
                    &input[edge.index]
                })
            ).collect::<anyhow::Result<Vec<_>>>()?;
            let output = node.operation.evaluate(inputs.as_slice())?;
            let new_step = InferenceStep {
                inputs: inputs.into_iter().cloned().collect(),
                outputs: output,
            };
            trace.new_step(node_id, new_step);
        }

        // compute the output tensor from the outputs of the output nodes
        let output_node = self.output_node();
        let output = trace.get_step(&output_node).ok_or(
         anyhow!("Output node {} not found in trace", output_node)   
        )?.outputs.outputs.clone();
        ensure!(output.len() == 1, "Unexpected number of output tensors");
        
        trace.output = output;

        Ok(trace)
    }

    pub(crate) fn provable_nodes(&self) -> impl Iterator<Item = (&NodeId, &ProvableNode<E, T, N>)> {
        self.nodes.iter().filter(|(_, node)|
            node.operation.is_provable()
        )
    }
}

#[cfg(test)]
mod tests {
    use goldilocks::GoldilocksExt2;
    use transcript::BasicTranscript;

    use crate::{layers::{activation::{Activation, Relu}, dense::Dense, provable::{Edge, OpInfo, ProvableNode}}, testing::random_vector, Element, Tensor};

    use super::ProvableModel;

    #[test]
    fn test_model_inference() {
        const INPUT_SIZE: usize = 45;
        let input_shape = vec![INPUT_SIZE];
        let mut model  = ProvableModel::<
            GoldilocksExt2, 
            BasicTranscript<GoldilocksExt2>,
            Element
        >::new_from_input_shapes(vec![input_shape.clone()]);
        // add input dense layer
        // generate random dense matrix
        let ncols = INPUT_SIZE;
        let nrows = 42;
        let dense = Dense::random(vec![nrows, ncols]);
        let dense_out_shape = &dense.output_shapes()[0];
        let input_node = model.add_node(ProvableNode::new(
            vec![Edge {
                node: None, // it's the input tensor for the model
                index: 0, // it's the first (and only) input tensor
            }],
            Box::new(dense),
        )).unwrap();
        // add activation layer
        let relu = Activation::Relu(Relu::new());
        let relu_node = model.add_node(ProvableNode::new(
           vec![Edge {
                node: Some(input_node),
                index: 0, // it's the first (and only) output tensor of the previous layer
            }],
            Box::new(relu),
        )).unwrap();
        // add another dense layer as output
        let nrows = 37;
        let ncols = dense_out_shape[0]; // it's a vector, so it has only one dimension
        let dense = Dense::random(vec![nrows, ncols]);
        let output_node = model.add_node(ProvableNode::new(
            vec![Edge {
                node: Some(relu_node),
                index: 0, // it's the first (and only) output tensor of the previous layer
            }],
            Box::new(dense)
        )).unwrap();

        let input = random_vector(INPUT_SIZE);
        let input_tensor = Tensor::new(input_shape.clone(), input);
        let trace = model.run(&[input_tensor]).unwrap();
        assert_eq!(trace.steps.len(), 3);
        assert_eq!(model.output_node(), output_node);
    }
}