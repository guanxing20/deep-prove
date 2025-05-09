use crate::{
    layers::provable::{
        Edge, NodeId, ProvableModel, ProvableNode, QuantizationStrategy, QuantizeOp, ToIterator,
    },
    padding::PaddingMode,
    quantization::{
        BIT_LEN,
        metadata::{MetadataBuilder, ModelMetadata},
    },
    tensor::Number,
};
use std::collections::HashMap;

use crate::{
    Element, Tensor,
    layers::{Layer, requant::Requant},
    model::Model,
    quantization,
};
use anyhow::{Result, anyhow, ensure};
use ark_std::rand;
use goldilocks::GoldilocksExt2;
use itertools::Itertools;
use statrs::statistics::{Data, Max, Min, OrderStatistics};
use tracing::{debug, info};

use super::ScalingFactor;

/// Trait for quantizing a float-based model into a quantized model. The current implementation
/// simply looks at the absolute maximum value of the model and uses that as the scaling factor
/// to quantize the model, one scaling factor per layer.
pub trait ScalingStrategy: std::fmt::Debug {
    fn quantize(
        &self,
        model: ProvableModel<f32>,
    ) -> Result<(ProvableModel<Element>, ModelMetadata)>;
    // {
    // ToDo: replace this implementation with an actual one
    // let input_not_padded_shape = model.unpadded_input_shape();
    // let model = ProvableModel::new_from_input_shapes(vec![input_not_padded_shape], PaddingMode::NoPadding);
    // let md = MetadataBuilder::new(ScalingFactor::default());
    // Ok((model, md.build()))
    // }

    fn name(&self) -> String;
}

/// Quantization strategy that observes the inference of the model with different inputs and uses the
/// min/max values of the output to determine the output scaling factor of each layer that needs
/// requantization afterwards.
#[derive(Debug)]
pub struct InferenceObserver {
    inputs: Vec<Vec<Vec<f32>>>,
}

impl InferenceObserver {
    pub fn new_with_representative_input(inputs: Vec<Vec<Vec<f32>>>) -> Self {
        Self { inputs }
    }
    pub fn new() -> Self {
        Self { inputs: vec![] }
    }
}

impl QuantizationStrategy for InferenceObserver {
    type AuxData = InferenceTracker;
}

const INPUT_TRACKING_ID: usize = 10_000;
impl ScalingStrategy for InferenceObserver {
    fn name(&self) -> String {
        format!("inference [{},{}]", *quantization::MIN, *quantization::MAX)
    }

    // x^2 + y^2 = AC^2
    // y^2 - b^2 = x^2 - a^2 = BM^2
    // a+b = AC -> a = AC -b
    // y^2 - b^2 = x^2 - (AC - b)^2 = x^2 - AC^2 - b^2 +2b*AC -> y^2 - x^2 = 2b*AC - AC^2 = 2b*AC - x^2 - y^2 -> 2*y^2 = 2b*AC
    // -> BM^2 + b^2 - b*AC = 0

    // b = (AC +- sqrt(AC^2 - 4BM^2))/2

    fn quantize(
        &self,
        model: ProvableModel<f32>,
    ) -> Result<(ProvableModel<Element>, ModelMetadata)> {
        let mut tracker = InferenceTracker::new();
        let input_shapes = model.input_shapes();
        let input_not_padded_shapes = model.unpadded_input_shapes();
        let inputs = if self.inputs.is_empty() {
            (0..10)
                .map(|_| {
                    input_shapes
                        .iter()
                        .map(|shape| {
                            let size = shape.iter().product();
                            (0..size)
                                .map(|_| <f32 as Number>::random(&mut rand::thread_rng()))
                                .collect_vec()
                        })
                        .collect_vec()
                })
                .collect()
        } else {
            self.inputs.clone()
        };
        // 1. Run the inference multiple times with different inputs
        // TODO: integrate that within model.rs in a more elegant way with inference step - currently problematic
        // because of the generics and FFT requirement to take a field
        let mut nsamples = 0;
        for input in inputs.into_iter() {
            let input_tensors = input
                .into_iter()
                .zip(model.unpadded_input_shapes())
                .enumerate()
                .map(|(i, (inp, shape))| {
                    let input_tensor = Tensor::new(shape, inp);
                    tracker.track(INPUT_TRACKING_ID as NodeId, i, input_tensor.clone());
                    input_tensor
                })
                .collect_vec();
            model.run_with_tracker::<GoldilocksExt2>(&input_tensors, Some(&mut tracker))?;
            nsamples += 1;
        }
        info!("InferenceObserver: {} samples observed", nsamples);
        // 2. get the scaling factor of the input
        let num_model_inputs = input_not_padded_shapes.len();
        let input_scaling = (0..num_model_inputs)
            .map(|i| {
                let (input_min, input_max) =
                    tracker.distribution_info(INPUT_TRACKING_ID as NodeId, i);
                ScalingFactor::from_absolute_max(input_min.abs().max(input_max.abs()), None)
            })
            .collect_vec();
        let mut md = MetadataBuilder::new(input_scaling);
        // 2. Create the requant layers from the infered data
        let mut requant_layers = vec![];
        let mut catch_err: Result<()> = Ok(());
        let nodes = model.intoto_forward_iterator_mut().map(|(node_id, node)| {
            let input_scaling = node.inputs.iter().map(|edge| {
                if let Some(n) = &edge.node {
                    let scalings = md.get_output_layer_scaling(n).ok_or(
                        anyhow!("Scaling factors for node {n} not found")
                    )?;
                    ensure!(edge.index < scalings.len(),
                        "Getting scaling factor {} for node {n}, but there are only {} scaling factors",
                        edge.index,
                        scalings.len(),
                    );
                    Ok(scalings[edge.index].clone())
                } else {
                    ensure!(edge.index < md.input_scaling.len(),
                        "Getting scaling factor {} for model inputs, but there are only {} scaling factors",
                        edge.index,
                        md.input_scaling.len(),
                    );
                    Ok(md.input_scaling[edge.index].clone())
                }
            }).collect::<Result<Vec<_>>>()?;
            let quantized_out = node.operation.quantize_op(&tracker, node_id, &input_scaling)?;
            md.set_layers_scaling(node_id, quantized_out.output_scalings, input_scaling);
            if let Some(requant) = quantized_out.requant_layer {
                requant_layers.push((node_id, requant));
            }
            let quantized_node = ProvableNode::new_with_outputs(
                node.inputs,
                quantized_out.quanzited_op,
                node.outputs,
            );
            Ok((node_id, quantized_node))
        }).map_while(|n|
            if n.is_err() {
                catch_err = Err(n.unwrap_err());
                None
            } else {
                Some(n.unwrap())
            }
        );
        let mut model =
            ProvableModel::new_from_shapes(input_not_padded_shapes, input_shapes, nodes);
        catch_err?;
        for (input_node_id, requant) in requant_layers {
            let node_id = model.add_requant_node(requant, input_node_id)?;
            // add scaling factor to `md` for requant layers: the scaling factors of the inputs correspond to
            // the scaling factors of the outputs of the previous node
            let input_scaling = md.get_output_layer_scaling(&input_node_id).ok_or(anyhow!(
                "Scaling factors not found for node {input_node_id}"
            ))?;
            let output_scaling = input_scaling.to_vec(); // output scaling factors are the same as input ones for requant
            md.set_layers_scaling(node_id, output_scaling, input_scaling.to_vec());
        }
        let out_node = model.output_node();
        let out_scaling = md
            .get_output_layer_scaling(&out_node)
            .ok_or(anyhow!(
                "Scaling factor for output node {out_node} not found"
            ))?
            .to_vec();
        Ok((model, md.build(out_scaling)))
    }
}

pub struct InferenceTracker {
    /// For each output of each node in the model of interest, we track all the values of the tensor
    data: HashMap<(NodeId, usize), Vec<f64>>,
}

impl InferenceTracker {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    pub(crate) fn track(&mut self, node_id: NodeId, output_index: usize, output: Tensor<f32>) {
        self.data
            .entry((node_id, output_index))
            .or_insert(Vec::new())
            .extend(output.get_data().iter().map(|x| *x as f64));
    }

    /// Returns the 0.05 and 0.95 quantiles of the distribution of the output values of the layer.
    pub(crate) fn distribution_info(&self, node_id: NodeId, output_index: usize) -> (f32, f32) {
        let mut d: Data<Vec<f64>> = Data::new(
            self.data
                .get(&(node_id, output_index))
                .expect(&format!(
                    "No data for output tensor {output_index} of node {node_id}"
                ))
                .clone(),
        );
        let min = d.percentile(5) as f32;
        let max = d.percentile(95) as f32;
        assert!(min <= max);
        //(min, max)
        (d.min() as f32, d.max() as f32)
        // let mean = d.mean().unwrap();
        // let std_dev = d.std_dev().unwrap();
        // let upper_bound = mean + 3.0 * std_dev;
        // let lower_bound = mean - 3.0 * std_dev;
        //(lower_bound as f32, upper_bound as f32)
    }
}

#[derive(Debug)]
pub struct AbsoluteMax(Option<Vec<f32>>);

impl AbsoluteMax {
    pub fn new_with_representative_input(input: Vec<f32>) -> Self {
        Self(Some(input))
    }
    pub fn new() -> Self {
        Self(None)
    }
}
impl ScalingStrategy for AbsoluteMax {
    fn name(&self) -> String {
        "absolute_max".to_string()
    }

    fn quantize(
        &self,
        model: ProvableModel<f32>,
    ) -> Result<(ProvableModel<Element>, ModelMetadata)> {
        todo!()
    }

    // fn quantize(&self, model: Model<f32>) -> Result<(Model<Element>, ModelMetadata)> {
    // let mut last_input_scaling_factor = if let Some(ref input) = self.0 {
    // let input_tensor = model.load_input_flat(input.clone());
    // ensure!(
    // model.input_shape() == input_tensor.get_shape(),
    // "input shape mismatch: expected {:?}, got {:?}",
    // model.input_shape(),
    // input_tensor.get_shape()
    // );
    // ScalingFactor::from_absolute_max(input_tensor.max_abs_output(), None)
    // } else {
    // ScalingFactor::default()
    // };
    // let mut md = MetadataBuilder::new(last_input_scaling_factor.clone());
    // let input_shape = model.input_shape();
    // let input_not_padded_shape = model.unpadded_input_shape();
    // let quantized_layers = model
    // .layers
    // .into_iter()
    // .enumerate()
    // .flat_map(|(id, l)| {
    // If a layer requires a requantization step the current layer, this method returns the
    // next layer, e.g. requantization layer, as well as the scaling factor of the output. This is
    // given to the next layer as input scaling factor.
    // match l {
    // Layer::Dense(d) => {
    // let max_weight = d.max_abs_weight();
    // let model_scaling = ScalingFactor::from_absolute_max(max_weight, None);
    // let bias_scaling = {
    // bias has to be quantized over integers with double bit length
    // let min_quantized = -(1 << (2 * (*BIT_LEN) - 1)) + 1;
    // let max_quantized = (1 << (2 * (*BIT_LEN) - 1)) - 1;
    // ScalingFactor::from_scale(
    // last_input_scaling_factor.scale() * model_scaling.scale(),
    // Some((min_quantized, max_quantized)),
    // )
    // };
    // let quantized_dense = d.quantize(&model_scaling, &bias_scaling);
    // let (quant_min_output, _quant_max_output) =
    // quantized_dense.output_range(*quantization::MIN, *quantization::MAX);
    // TODO: remove this is broken
    // let output_scaling = ScalingFactor::default();
    // last_input_scaling_factor = output_scaling;
    // md.set_layers_scaling(id, output_scaling);
    // let shift =
    // last_input_scaling_factor.shift(&model_scaling, &output_scaling);
    // let requant = Requant::new(quant_min_output.abs() as usize, shift);
    // vec![Layer::Dense(quantized_dense), Layer::Requant(requant)]
    // }
    // Layer::Convolution(d) => {
    // let max_weight = d.max_abs_weight();
    // let model_scaling = ScalingFactor::from_absolute_max(max_weight, None);
    // let bias_scaling = {
    // bias has to be quantized over integers with double bit length
    // let min_quantized = -(1 << (2 * (*BIT_LEN) - 1)) + 1;
    // let max_quantized = (1 << (2 * (*BIT_LEN) - 1)) - 1;
    // ScalingFactor::from_scale(
    // last_input_scaling_factor.scale() * model_scaling.scale(),
    // Some((min_quantized, max_quantized)),
    // )
    // };
    // let quantized_conv = d.quantize(&model_scaling, &bias_scaling);
    // let (quant_min_output, _quant_max_output) =
    // quantized_conv.output_range(*quantization::MIN, *quantization::MAX);
    // TODO: remove this is broken
    // let output_scaling = ScalingFactor::default();
    // md.set_layers_scaling(id, output_scaling);
    // let shift =
    // last_input_scaling_factor.shift(&model_scaling, &output_scaling);
    // last_input_scaling_factor = output_scaling;
    // let requant = Requant::new(quant_min_output.abs() as usize, shift);
    // vec![Layer::Convolution(quantized_conv), Layer::Requant(requant)]
    // }
    // a => {
    // return vec![a.quantize(
    // &last_input_scaling_factor,
    // None, // no scaling factor for bias needed for this layer
    // )];
    // }
    // }
    // })
    // .collect::<Vec<Layer<Element>>>();
    // Ok((
    // Model::<Element>::new_from(quantized_layers, input_not_padded_shape, input_shape),
    // md.build(),
    // ))
    // }
}
