use std::collections::HashMap;

use crate::{
    layers::{requant::Requant, Layer, LayerOutput}, model::Model, quantization, Element, Tensor
};
use statrs::statistics::{Data, OrderStatistics};
use anyhow::{bail, ensure, Result};
use tracing::{info, warn};

use super::ScalingFactor;

/// Trait for quantizing a float-based model into a quantized model. The current implementation
/// simply looks at the absolute maximum value of the model and uses that as the scaling factor
/// to quantize the model, one scaling factor per layer.
pub trait ScalingStrategy: std::fmt::Debug {
    fn quantize(&self, model: Model<f32>) -> Result<Model<Element>>;
}

#[derive(Debug)]
pub struct InferenceObserver {
    inputs: Vec<Tensor<f32>>,
}

const INPUT_TRACKING_ID: usize = 10_000;
impl ScalingStrategy for InferenceObserver {
    fn quantize(&self, model: Model<f32>) -> Result<Model<Element>> {
        let mut tracker = InferenceTracker::new();
        let input_shape = model.input_shape();
        let input_not_padded_shape = model.input_not_padded();
        // 1. Run the inference multiple times with different inputs
        // TODO: integrate that within model.rs in a more elegant way with inference step - currently problematic
        // because of the generics and FFT requirement to take a field
        for (i, input) in self.inputs.iter().enumerate() {
            ensure!(
                model.input_shape() == input.get_shape(),
                "input shape mismatch: expected {:?}, got {:?}",
                model.input_shape(),
                input.get_shape()
            );
            let prepared_input = model.prepare_input(input.clone());
            let mut last_output = prepared_input;
            tracker.track(INPUT_TRACKING_ID, last_output.clone());
            for (id, layer) in model.layers.iter().enumerate() {
                info!("Inference Observer: inference run #{}: running layer {}",i,layer.describe());
                last_output= run_layer(layer, &last_output);
                tracker.track(id, last_output.clone());
            }
        }
        // 2. get the scaling factor of the input
        let (input_min,input_max) = tracker.get_bounds(INPUT_TRACKING_ID);
        let last_input_scaling = ScalingFactor::new(input_min.abs().max(input_max.abs()));
        // 2. Create the requant layers from the infered data
        let quantized_layers = model.layers.into_iter().enumerate().map(|(id, layer)| {
            match layer {
                Layer::Dense(ref dense) => {
                    let (min, max) = tracker.get_bounds(id);
                    // since we are doing symmetric scaling
                    let abs_max = min.abs().max(max.abs());
                    let model_scaling = ScalingFactor::new(dense.max_abs_weight());
                    let requant = requant_from(last_input_scaling, model_scaling, abs_max);
                    vec![layer.quantize(model_scaling), requant]
                }
                Layer::Convolution(ref conv) => {
                    let (min, max) = tracker.get_bounds(id);
                    let abs_max = min.abs().max(max.abs());
                    let model_scaling = ScalingFactor::new(conv.max_abs_weight());
                    let requant = requant_from(last_input_scaling, model_scaling, abs_max);
                    vec![layer.quantize(model_scaling), requant]
                }
                _ => vec![]
            }
        }).flatten().collect::<Vec<_>>();
        Ok(Model::<Element>::new_from(
            quantized_layers,
            input_not_padded_shape,
            input_shape,
        ))
    }
}
    fn run_layer(layer: &Layer<f32>,input: &Tensor<f32>) -> Tensor<f32> {
        match layer {
            Layer::Dense(ref dense) => dense.op(input),
            Layer::Activation(activation) => activation.op(input),
            Layer::Convolution(ref conv_pair) => 
                input.conv2d(&conv_pair.filter, &conv_pair.bias, 1),
            Layer::Pooling(info) => info.op(input),
            // Traditional convolution is used for debug purposes. That is because the actual convolution
            // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
            Layer::SchoolBookConvolution(_) => {
                panic!("InferenceObserver: schoolbook convolution found while observing inference on float !?");
            }
            Layer::Requant(_) => {
                panic!("InferenceObserver: requantization layer found while observing inference on float !?");
            }
        }
    }

struct InferenceTracker {
    /// For each layer of interest, we track all the outputs of that layer
    data: HashMap<usize, Vec<f64>>,
}

impl InferenceTracker {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    fn track(&mut self, layer_index: usize, output: Tensor<f32>) {
        self.data.entry(layer_index).or_insert(Vec::new()).extend(output.get_data().iter().map(|x| *x as f64));
    }

    fn get_bounds(&self, layer_index: usize) -> (f32, f32) {
        let mut d : Data<Vec<f64>> = Data::new(self.data[&layer_index].clone());
        let min = d.percentile(5) as f32;
        let max = d.percentile(95) as f32;
        (min, max)
    }
}

#[derive(Debug)]
pub struct AbsoluteMax(Option<Tensor<f32>>);

impl AbsoluteMax {
    pub fn new_with_representative_input(input: Tensor<f32>) -> Self {
        Self(Some(input))
    }
    pub fn new() -> Self {
        Self(None)
    }
}
impl ScalingStrategy for AbsoluteMax {
    fn quantize(&self, model: Model<f32>) -> Result<Model<Element>> {
        let mut last_input_scaling_factor = if let Some(ref input) = self.0 {
            ensure!(
                model.input_shape() == input.get_shape(),
                "input shape mismatch: expected {:?}, got {:?}",
                model.input_shape(),
                input.get_shape()
            );
            let prepared_input = model.prepare_input(input.clone());
            ScalingFactor::new(prepared_input.max_abs_output())
        } else {
            ScalingFactor::default()
        };
        let input_shape = model.input_shape();
        let input_not_padded_shape = model.input_not_padded();
        let quantized_layers = model
            .layers
            .into_iter()
            .flat_map(|l| {
                // If a layer requires a requantization step the current layer, this method returns the
                // next layer, e.g. requantization layer, as well as the scaling factor of the output. This is
                // given to the next layer as input scaling factor.
                match l {
                    Layer::Dense(ref d) => {
                        let max_weight = d.max_abs_weight();
                        let model_scaling = ScalingFactor::new(max_weight);
                        let max_output = d.max_abs_output(last_input_scaling_factor);
                        let output_scaling = ScalingFactor::new(max_output);
                        last_input_scaling_factor = output_scaling;
                        println!("Scaling: AbsoluteMax: DENSE max_weight {:?}, max_output: {:?} - adding requant",max_weight,max_output);
                        vec![
                            l.quantize(model_scaling),
                            requant_from(last_input_scaling_factor, model_scaling, max_output),
                        ]
                    }
                    Layer::Convolution(ref d) => {
                        let max_weight = d.max_abs_weight();
                        let model_scaling = ScalingFactor::new(max_weight);
                        let max_output = d.max_abs_output(last_input_scaling_factor);
                        let output_scaling = ScalingFactor::new(max_output);
                        last_input_scaling_factor = output_scaling;
                        println!("Scaling: AbsoluteMax: CONV max_weight {:?}, max_output: {:?} - adding requant", max_weight, max_output);
                        vec![
                            l.quantize(model_scaling),
                            requant_from(last_input_scaling_factor, model_scaling, max_output),
                        ]
                    }
                    a => return vec![a.quantize(last_input_scaling_factor)],
                }
            })
            .collect::<Vec<Layer<Element>>>();
        Ok(Model::<Element>::new_from(
            quantized_layers,
            input_not_padded_shape,
            input_shape,
        ))
    }
}

fn requant_from(s1: ScalingFactor, s2: ScalingFactor, max_output: f32) -> Layer<Element> {
    let s3 = ScalingFactor::new(max_output);
    let shift = s1.shift(&s2, &s3);
    let quantized_output = s3.quantize(&max_output);
    /// requantized_output = S1 * S2 / S3 * MODEL * INPUT
    /// we want maximum NON-QUANTIZED utput, so range = MODEL * INPUT, e.g. it approximates the 
    /// maximum absolute value of the non-requantized output
    /// so range = requantized_output * S3 / (S1 * S2)
    /// NOTE: a better approximation would be to actually run the quantized model and check the maximum absolute value of the output.
    /// This is however costly, so we use this approximation for now.
    let range = (quantized_output as f32 * s3.scale() / (s1.scale() * s2.scale())).ceil() as usize;
    Layer::Requant(Requant {
        right_shift: shift,
        range: range,
        after_range: 1 << *quantization::BIT_LEN,
    })
}

