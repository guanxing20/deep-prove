mod error;
mod model;

use ff_ext::ExtensionField;
use serde::{de::DeserializeOwned, Serialize};
use transcript::Transcript;

use crate::{commit::precommit::PolyID, iop::{context::ContextAux, verifier::Verifier}, lookup::context::LookupWitnessGen, tensor::{ConvData, Number}, Claim, Element, Prover, Tensor};

use super::{LayerCtx, LayerProof};

pub(crate) type NodeId = u64;

pub use error::ProvableOpError;
pub use model::{ProvableModel, InferenceTrace};

/// Represents a link between an input/output wire of a node with an input/output wire of
/// another node.
#[derive(Clone, Debug, Default)]
pub struct Edge {
    // Reference to the node linked to this wire, will be `None` if the wire is an input or
    // output of the model
    pub(crate) node: Option<NodeId>,
    // The index of the wire of `node` which is linked to this wire
    pub(crate) index: usize,
}

/// Represents all the edges that are connected to a node's output wire
#[derive(Clone, Debug, Default)]
pub struct OutputWire {
    // needs to be a vector because the output of a node can be used as input to multiple nodes
    pub(crate) edges: Vec<Edge>,
}

#[derive(Clone, Debug)]
pub struct Node<Op> {
    pub(crate) inputs: Vec<Edge>,
    pub(crate) outputs: Vec<OutputWire>,
    pub(crate) operation: Op,
}

pub type ProvableNode<E, T, D> = Node<Box<dyn ProvableOp<E, T, D>>>;

impl<E, T, D> ProvableNode<E, T, D> {
    pub(crate) fn new(
        inputs: Vec<Edge>,
        operation: Box<dyn ProvableOp<E, T, D>>,
    ) -> Self {
        let num_inputs = inputs.len();
        Self {
            inputs,
            outputs: vec![Default::default(); operation.num_outputs(num_inputs)],
            operation,
        }
    }
}

/*pub struct ProvableNode<E, T, D> {
    inputs: Vec<Edge>,
    input_shape: Vec<usize>,
    outputs: Vec<OutputWire>,
    output_shape: Vec<usize>,
    pub(crate) operation: Box<dyn ProvableOp<E, T, D>>,
}*/

pub struct LayerOut<T, E: ExtensionField> {
    pub(crate) outputs: Vec<Tensor<T>>,
    pub(crate) proving_data: Option<ConvData<E>>,
}

impl<T, E: ExtensionField> LayerOut<T, E> {
    pub(crate) fn from_vec(out: Vec<Tensor<T>>) -> Self {
        Self {
            outputs: out,
            proving_data: None,
        }
    }

    pub fn outputs(&self) -> Vec<&Tensor<T>> {
        self.outputs.iter().collect()
    }
}

pub trait OpInfo {
    /// Returns the expected input shapes (in input order)
    fn input_shapes(&self) -> Vec<Vec<usize>>;
        
    /// Returns the shapes of the outputs (in the same order)
    fn output_shapes(&self) -> Vec<Vec<usize>>;

    fn num_outputs(&self, num_inputs: usize) -> usize;

    fn describe(&self) -> String;
}

pub trait Op<T: Number, E: ExtensionField>: OpInfo {
    /// Evaluates the operation given any inputs tensors and constant inputs.
    fn evaluate(&self, inputs: &[&Tensor<T>]) -> Result<LayerOut<T, E>, ProvableOpError>;
}


pub(crate) fn evaluate_layer<E: ExtensionField, T: Number, O: Op<T, E>>(
        layer: &O, 
        inputs: &[&Tensor<T>]
    ) -> Result<LayerOut<T, E>, ProvableOpError> {
        layer.evaluate(inputs) 
    }

pub trait ProveInfo<E: ExtensionField> 
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(
        &self,
        id: PolyID,
        aux: ContextAux,
    ) -> (LayerCtx<E>, ContextAux);

    fn commit_info(&self, _id: NodeId) -> Vec<Option<(PolyID, Vec<E>)>> {
        vec![None]
    }
}

pub trait ProvableOp<E, T, N>: Op<N, E> + ProveInfo<E>
where 
            E: ExtensionField,
            E::BaseField: Serialize + DeserializeOwned,
            E: Serialize + DeserializeOwned,
            T: Transcript<E>,
            N: Number,
{
    /// Returns the inputs shapes padded for proving
    fn padded_input_shapes(&self) -> Vec<Vec<usize>> {
        self.input_shapes().iter().map(|shape|
            shape.iter().map(|dim|
                dim.next_power_of_two()
            ).collect()
        ).collect()
    }

    /// Returns the outputs shapes padded for proving
    fn padded_output_shapes(&self) -> Vec<Vec<usize>> {
        self.output_shapes().iter().map(|shape|
            shape.iter().map(|dim|
                dim.next_power_of_two()
            ).collect()
        ).collect()
    }

    fn is_provable(&self) -> bool;
    
    /// Produces a proof of correct execution for this operation.
    fn prove(&self, ctx: &LayerCtx<E>, last_claims: Vec<Claim<E>>, step_data: &InferenceStep<E, E>, prover: &mut Prover<E, T>) -> Result<Vec<Claim<E>>, ProvableOpError> {
        // Default implementation, to avoid having to implement this method in case `is_provable` is false
        assert!(!self.is_provable(), "Running default prove implementation for a provable operation! Implement prove method");
        Ok(vec![Claim::default()])
    }
    
    /// Verifies a proof for this operation type
    fn verify(&self, proof: &LayerProof<E>, ctx: LayerCtx<E>, last_claims: Vec<Claim<E>>, verifier: &mut Verifier<E, T>) -> Result<Vec<Claim<E>>, ProvableOpError> {
        // Default implementation, to avoid having to implement this method in case `is_provable` is false
        assert!(!self.is_provable(), "Running default verify implementation for a provable operation! Implement verify method");
        Ok(vec![Claim::default()])
    }

    fn gen_lookup_witness(&self, id: NodeId, gen: &mut LookupWitnessGen<E>, step_data: &InferenceStep<Element, E>) -> Result<(), ProvableOpError> {
        Ok(())
    }
}

pub struct InferenceStep<N, E: ExtensionField> {
    pub(crate) inputs: Vec<Tensor<N>>,
    pub(crate) outputs: LayerOut<N, E>,
}