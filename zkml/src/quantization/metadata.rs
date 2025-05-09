//! Metadata related information for a model. These are the information derived from the
//! float based model weights and activations.
use std::collections::HashMap;

use crate::{layers::provable::NodeId, model::Model};

use super::ScalingFactor;

/// Structure holding the scaling factors of the input and output of each layer
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub input: Vec<ScalingFactor>,
    pub(crate) input_layers_scaling: HashMap<NodeId, Vec<ScalingFactor>>,
    pub(crate) output_layers_scaling: HashMap<NodeId, Vec<ScalingFactor>>,
    pub(crate) output: Vec<ScalingFactor>,
    pub float_model: Option<Model<f32>>,
}

impl ModelMetadata {
    pub fn output_scaling_factor(&self) -> Vec<ScalingFactor> {
        self.output.clone()
    }

    pub fn layer_output_scaling_factor(&self, node_id: NodeId) -> &[ScalingFactor] {
        self.output_layers_scaling
            .get(&node_id)
            .expect(&format!("Node {node_id} not found"))
    }

    pub fn layer_input_scaling_factor(&self, node_id: NodeId) -> &[ScalingFactor] {
        self.input_layers_scaling
            .get(&node_id)
            .expect(&format!("Node {node_id} not found"))
    }
}

pub(crate) struct MetadataBuilder {
    pub(crate) input_scaling: Vec<ScalingFactor>,
    output_layers_scaling: HashMap<NodeId, Vec<ScalingFactor>>,
    input_layers_scaling: HashMap<NodeId, Vec<ScalingFactor>>,
}

impl MetadataBuilder {
    pub fn new(input_scaling: Vec<ScalingFactor>) -> Self {
        Self {
            input_scaling,
            output_layers_scaling: HashMap::new(),
            input_layers_scaling: HashMap::new(),
        }
    }

    pub fn set_layers_scaling(
        &mut self,
        node_id: NodeId,
        output_scaling: Vec<ScalingFactor>,
        input_scaling: Vec<ScalingFactor>,
    ) {
        self.output_layers_scaling.insert(node_id, output_scaling);
        self.input_layers_scaling.insert(node_id, input_scaling);
    }

    pub(crate) fn get_output_layer_scaling(&self, node_id: &NodeId) -> Option<&[ScalingFactor]> {
        self.output_layers_scaling
            .get(node_id)
            .map(|s| s.as_slice())
    }

    pub fn build(self, output_scaling: Vec<ScalingFactor>) -> ModelMetadata {
        ModelMetadata {
            input: self.input_scaling,
            output_layers_scaling: self.output_layers_scaling,
            input_layers_scaling: self.input_layers_scaling,
            output: output_scaling,
            float_model: None,
        }
    }
}
