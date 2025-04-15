//! Metadata related information for a model. These are the information derived from the
//! float based model weights and activations.
use std::collections::HashMap;

use crate::model::Model;

use super::ScalingFactor;

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub input: ScalingFactor,
    pub(crate) output_layers_scaling: HashMap<usize, ScalingFactor>,
    pub float_model: Option<Model<f32>>,
}

impl ModelMetadata {
    pub fn output_scaling_factor(&self) -> ScalingFactor {
        self.output_layers_scaling[self.output_layers_scaling.keys().max().unwrap()].clone()
    }
    pub fn layer_output_scaling_factor(&self, layer_id: usize) -> ScalingFactor {
        self.output_layers_scaling.get(&layer_id).expect(&format!("Layer {} not found", layer_id)).clone()
    }
}

pub(crate) struct MetadataBuilder {
    input_scaling: Option<ScalingFactor>,
    layers_scaling: HashMap<usize, ScalingFactor>,
}

impl MetadataBuilder {
    pub fn new() -> Self {
        Self {
            input_scaling: None,
            layers_scaling: HashMap::new(),
        }
    }

    pub fn set_input_scaling(&mut self, input_scaling: ScalingFactor) {
        self.input_scaling = Some(input_scaling);
    }

    pub fn set_layers_scaling(&mut self, layer_id: usize, output_scaling: ScalingFactor) {
        self.layers_scaling.insert(layer_id, output_scaling);
    }

    pub fn build(self) -> ModelMetadata {
        ModelMetadata {
            input: self.input_scaling.unwrap(),
            output_layers_scaling: self.layers_scaling,
            float_model: None,
        }
    }
}
