use crate::{
    Element, Tensor,
    layers::{Layer, requant::Requant},
    model::Model,
    quantization,
};
use anyhow::{Result, ensure};

use super::ScalingFactor;

/// Trait for quantizing a float-based model into a quantized model. The current implementation
/// simply looks at the absolute maximum value of the model and uses that as the scaling factor
/// to quantize the model, one scaling factor per layer.
pub trait ScalingStrategy: std::fmt::Debug {
    fn quantize(&self, model: Model<f32>) -> Result<Model<Element>>;
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

