pub mod activation;
pub mod common;
pub mod convolution;
pub mod dense;
pub mod hadamard;
pub mod matvec;
pub mod pooling;
pub mod requant;
pub mod reshape;
pub mod provable;

use std::fmt::Debug;

use anyhow::Result;
use ff_ext::ExtensionField;
use goldilocks::GoldilocksExt2;
use itertools::Itertools;
use pooling::{PoolingCtx, PoolingProof};
use provable::{evaluate_layer, LayerOut, Op, OpInfo, ProvableNode, ProvableOpError, ProveInfo};
use requant::RequantCtx;
use reshape::Reshape;
use statrs::statistics::{Data, Distribution};
use transcript::Transcript;
use tracing::debug;

use crate::{
    Element,
    commit::precommit::PolyID,
    iop::context::{ContextAux, ShapeStep, TableCtx},
    layers::{
        activation::{Activation, ActivationProof, Relu},
        convolution::Convolution,
        dense::Dense,
        pooling::Pooling,
        requant::{Requant, RequantProof},
    },
    padding::PaddingMode,
    quantization::ScalingFactor,
    tensor::{ConvData, Number, Tensor},
};
use activation::ActivationCtx;
use convolution::{ConvCtx, ConvProof, SchoolBookConvCtx};
use dense::{DenseCtx, DenseProof};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Clone, Debug)]
pub enum Layer<T> {
    Dense(Dense<T>),
    // TODO: replace this with a Tensor based implementation
    Convolution(Convolution<T>),
    // Traditional convolution is used for debug purposes. That is because the actual convolution
    // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
    SchoolBookConvolution(Convolution<T>),
    Activation(Activation),
    // this is the output quant info. Since we always do a requant layer after each dense,
    // then we assume the inputs requant info are default()
    Requant(Requant),
    Pooling(Pooling),
    // TODO: so far it's only flattening the input tensor, e.g. new_shape = vec![shape.iter().product()]
    Reshape(Reshape),
}

/// Describes a steps wrt the polynomial to be proven/looked at. Verifier needs to know
/// the sequence of steps and the type of each step from the setup phase so it can make sure the prover is not
/// cheating on this.
/// NOTE: The context automatically appends a requant step after each dense layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub enum LayerCtx<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    Dense(DenseCtx<E>),
    Convolution(ConvCtx<E>),
    SchoolBookConvolution(SchoolBookConvCtx),
    Activation(ActivationCtx),
    Requant(RequantCtx),
    Pooling(PoolingCtx),
    Table(TableCtx<E>),
    Reshape,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum LayerProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    Dense(DenseProof<E>),
    Convolution(ConvProof<E>),
    Activation(ActivationProof<E>),
    Requant(RequantProof<E>),
    Pooling(PoolingProof<E>),
    Reshape,
}
#[derive(Clone, Debug)]
pub enum LayerOutput<F>
where
    F: ExtensionField,
{
    NormalOut(Tensor<Element>),
    ConvOut((Tensor<Element>, ConvData<F>)),
}

impl<E> LayerCtx<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::SchoolBookConvolution(_) => "Traditional Convolution".to_string(),
            Self::Convolution(_) => "Convolution".to_string(),
            Self::Activation(_) => "Activation".to_string(),
            Self::Requant(_) => "Requant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
            Self::Table(..) => "Table".to_string(),
            Self::Reshape => "Reshape".to_string(),
        }
    }

    // TODO: is this used and correct ??
    pub fn requires_lookup(&self) -> bool {
        match self {
            Self::Dense(..) => false,
            Self::Reshape => false,
            _ => true,
        }
    }
    pub fn output_shape(&self, input_shape: &[usize], padding_mode: PaddingMode) -> Vec<usize> {
        match self {
            Self::Dense(ref dense) => dense.output_shape(input_shape, padding_mode),
            Self::Convolution(ref filter) => filter.output_shape(input_shape, padding_mode),
            Self::SchoolBookConvolution(ref _filter) => {
                panic!("SchoolBookConvolution should NOT be used in proving")
            }
            Self::Activation(..) => input_shape.to_vec(),
            Self::Requant(..) => input_shape.to_vec(),
            Self::Pooling(ref pooling) => pooling.output_shape(input_shape),
            Self::Reshape => <Reshape as OpInfo>::output_shapes(&Reshape, &vec![input_shape.to_vec()], padding_mode)[0].clone(),
            Self::Table(..) => panic!("Table should NOT be used in proving"),
        }
    }
    pub fn next_shape_step(&self, last_step: &ShapeStep) -> ShapeStep {
        let unpadded_output = last_step.unpadded_output_shape.iter().map(|shape|
            self.output_shape(&shape, PaddingMode::NoPadding)
        ).collect();
        let padded_output = last_step.padded_output_shape.iter().map(|shape|
            self.output_shape(&shape, PaddingMode::Padding)
        ).collect();
        ShapeStep::next_step(last_step, unpadded_output, padded_output)
    }
    pub fn shape_step(&self, unpadded_input: &[Vec<usize>], padded_input: &[Vec<usize>]) -> ShapeStep {
        let unpadded_output = unpadded_input.iter().map(|shape|
            self.output_shape(&shape, PaddingMode::NoPadding)
        ).collect();
        let padded_output = padded_input.iter().map(|shape| 
            self.output_shape(&shape, PaddingMode::Padding)
        ).collect();
        ShapeStep::new(
            unpadded_input.to_vec(),
            padded_input.to_vec(),
            unpadded_output,
            padded_output,
        )
    }
}

impl<E, T, N> ProvableNode<E, T, N> 
where 
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    T: Transcript<E>,
    N: Number,
{
    pub fn describe(&self) -> String {
        self.operation.describe()
    }

    pub fn is_provable(&self) -> bool {
        self.operation.is_provable()
    }

    pub(crate) fn run(&self, inputs: &[&Tensor<N>], unpadded_input_shapes: Vec<Vec<usize>>) -> Result<LayerOut<N, E>, ProvableOpError> 
    where 
        E: ExtensionField,
        N: Number,
    {
        self.operation.evaluate(inputs, unpadded_input_shapes)
    }
     
    pub(crate) fn step_info(
        &self,
        id: PolyID,
        aux: ContextAux
    ) -> (LayerCtx<E>, ContextAux) {
        self.operation.step_info(id, aux)
    }
}

impl<T: Number> Layer<T> {
    pub fn output_shape(&self, input_shape: &[usize], padding_mode: PaddingMode) -> Vec<usize> {
        match self {
            Layer::Dense(ref dense) => dense.output_shape(input_shape, padding_mode),
            Layer::Convolution(ref filter) => filter.output_shape(input_shape, padding_mode),
            Layer::SchoolBookConvolution(ref filter) => {
                filter.output_shape(input_shape, padding_mode)
            }
            Layer::Activation(Activation::Relu(_)) => input_shape.to_vec(),
            Layer::Requant(_) => input_shape.to_vec(),
            Layer::Pooling(Pooling::Maxpool2D(info)) => info.output_shape(input_shape),
            Layer::Reshape(ref r) => r.output_shapes(&vec![input_shape.to_vec()], padding_mode)[0].clone(),
        }
    }
    /// Returns the shape of the layer as used in the model. If the layer do NOT have a shape per se,
    /// e.g. RELU for example, it returns None.
    pub fn model_shape(&self) -> Option<Vec<usize>> {
        match &self {
            Layer::Dense(ref dense) => Some(dense.matrix.get_shape()),

            Layer::Convolution(ref filter) => Some(filter.get_shape()),
            Layer::SchoolBookConvolution(ref filter) => Some(filter.get_shape()),

            Layer::Activation(Activation::Relu(_)) => None,
            Layer::Requant(_) => None,
            Layer::Pooling(Pooling::Maxpool2D(info)) => {
                Some(vec![info.kernel_size, info.kernel_size])
            }
            Layer::Reshape(ref _reshape) => None,
        }
    }

    pub fn describe(&self) -> String {
        match &self {
            Layer::Dense(ref dense) => dense.describe(),
            Layer::Convolution(ref filter) => {
                format!(
                    "Conv: ({},{},{},{})",
                    filter.kw(),
                    filter.kx(),
                    filter.nw(),
                    filter.nw()
                )
            }
            Layer::SchoolBookConvolution(ref _filter) => {
                format!(
                    "Conv: Traditional convolution for debug purposes" /* matrix.fmt_integer() */
                )
            }
            Layer::Activation(Activation::Relu(_)) => {
                format!("RELU: {}", 1 << Relu::num_vars())
            }
            Layer::Requant(info) => {
                format!(
                    "Requant: shift: {}, offset: 2^{}",
                    info.right_shift,
                    (info.range << 1).ilog2() as usize,
                )
            }
            Layer::Pooling(Pooling::Maxpool2D(info)) => format!(
                "MaxPool2D{{ kernel size: {}, stride: {} }}",
                info.kernel_size, info.stride
            ),
            Layer::Reshape(ref reshape) => reshape.describe(),
        }
    }
    pub fn needs_requant(&self) -> bool {
        match self {
            Layer::Dense(..) | Layer::Convolution(..) => true,
            _ => false,
        }
    }
    pub fn is_provable(&self) -> bool {
        match self {
            Layer::Reshape(..) => false,
            _ => true,
        }
    }
}

impl Layer<f32> {
    pub fn quantize(self, s: &ScalingFactor, bias_s: Option<&ScalingFactor>) -> Layer<Element> {
        match self {
            Layer::Dense(dense) => {
                Layer::Dense(dense.quantize(s, bias_s.expect("bias_s is required for dense layer")))
            }
            Layer::Convolution(conv) => Layer::Convolution(conv.quantize(
                &s,
                bias_s.expect("bias_s is required for convolution layer"),
            )),
            Layer::SchoolBookConvolution(conv) => Layer::SchoolBookConvolution(conv.quantize(
                &s,
                bias_s.expect("bias_s is required for schoolbook convolution layer"),
            )),
            Layer::Activation(activation) => Layer::Activation(activation),
            Layer::Requant(requant) => Layer::Requant(requant),
            Layer::Pooling(pooling) => Layer::Pooling(pooling),
            Layer::Reshape(_reshape) => Layer::Reshape(Reshape),
        }
    }
    /// TODO: limitation of enum is we can't have same names as in Element run
    pub(crate) fn run(&self, input: &Tensor<f32>) -> Tensor<f32> {
        match self {
            Layer::Dense(ref dense) => evaluate_layer::<GoldilocksExt2, _, _>(dense, &vec![input], None).unwrap().outputs()[0].clone(),
            Layer::Activation(activation) => evaluate_layer::<GoldilocksExt2, _, _>(activation, &vec![input], None).unwrap().outputs()[0].clone(),
            Layer::Convolution(ref conv_pair) => {
                input.conv2d(&conv_pair.filter, &conv_pair.bias, 1)
            }
            Layer::Pooling(info) => info.op(input),
            // Traditional convolution is used for debug purposes. That is because the actual convolution
            // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
            Layer::SchoolBookConvolution(ref conv_pair) => {
                input.conv2d(&conv_pair.filter, &conv_pair.bias, 1)
            }
            Layer::Reshape(ref reshape) => evaluate_layer::<GoldilocksExt2, _, _>(reshape, &vec![input], None).unwrap().outputs()[0].clone(),
            Layer::Requant(_) => {
                panic!(
                    "InferenceObserver: requantization layer found while observing inference on float !?"
                );
            }
        }
    }
}

impl Layer<Element> {
    pub(crate) fn step_info<E>(&self, id: PolyID, aux: ContextAux) -> (LayerCtx<E>, ContextAux)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        match self {
            Layer::Dense(dense) => dense.step_info(id, aux),
            Layer::Convolution(conv) => conv.step_info(id, aux),
            Layer::SchoolBookConvolution(_conv) => SchoolBookConvCtx.step_info(id, aux),
            Layer::Activation(activation) => activation.step_info(id, aux),
            Layer::Requant(requant) => requant.step_info(id, aux),
            Layer::Pooling(pooling) => pooling.step_info(id, aux),
            Layer::Reshape(reshape) => reshape.step_info(id, aux),
        }
    }

    /// Run the operation associated with that layer with the given input
    // TODO: move to tensor library : right now it works because we assume there is only Dense
    // layer which is matmul
    pub fn op<F: ExtensionField>(
        &self,
        input: &Tensor<Element>,
        unpadded_shape: &[usize],
    ) -> Result<LayerOutput<F>> {
        let output = match &self {
            Layer::Dense(ref dense) => Ok(LayerOutput::NormalOut(evaluate_layer::<GoldilocksExt2, _, _>(dense, &vec![input], Some(vec![unpadded_shape.to_vec()])).unwrap().outputs()[0].clone())),
            Layer::Activation(activation) => Ok(LayerOutput::NormalOut(evaluate_layer::<GoldilocksExt2, _, _>(activation, &vec![input], Some(vec![unpadded_shape.to_vec()])).unwrap().outputs()[0].clone())),
            Layer::Convolution(ref filter) => {
                Ok(LayerOutput::ConvOut(filter.op(input, unpadded_shape)))
            }
            // Traditional convolution is used for debug purposes. That is because the actual convolution
            // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
            Layer::SchoolBookConvolution(ref conv_pair) => Ok(LayerOutput::NormalOut(
                input.conv2d(&conv_pair.filter, &conv_pair.bias, 1),
            )),

            Layer::Requant(info) => info.op(input).map(|r| LayerOutput::NormalOut(r)),
            Layer::Pooling(info) => Ok(LayerOutput::NormalOut(info.op(input))),
            Layer::Reshape(reshape) => Ok(LayerOutput::NormalOut(evaluate_layer::<GoldilocksExt2, _, _>(reshape, &vec![input], Some(vec![unpadded_shape.to_vec()])).unwrap().outputs()[0].clone())),
        }?;
        match output {
            LayerOutput::NormalOut(ref output) => {
                debug!(
                    "Layer::{:?}: shape {:?} op: {:?} - min {:?}, max {:?}",
                    self.describe(),
                    output.get_shape(),
                    &output.get_data()[..output.get_data().len().min(10)],
                    output.get_data().iter().min().unwrap(),
                    output.get_data().iter().max().unwrap()
                );
            }
            LayerOutput::ConvOut((ref output, _)) => {
                let d = Data::new(output.get_data().iter().map(|e| *e as f64).collect_vec());
                debug!(
                    "Layer::{:?}: shape {:?} op: {:?} - min {:?}, max {:?}, mean {:?}, std {:?}",
                    self.describe(),
                    output.get_shape(),
                    &output.get_data()[..output.get_data().len().min(10)],
                    output.get_data().iter().min().unwrap(),
                    output.get_data().iter().max().unwrap(),
                    d.mean().unwrap(),
                    d.std_dev().unwrap()
                );
            }
        }
        Ok(output)
    }
}

impl<E: ExtensionField> LayerProof<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::Convolution(_) => "Convolution".to_string(),
            Self::Activation(_) => "Activation".to_string(),
            Self::Requant(_) => "Requant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
            Self::Reshape => "Reshape".to_string(),
        }
    }

    pub fn get_lookup_data(&self) -> Option<(Vec<E>, Vec<E>)> {
        match self {
            LayerProof::Dense(..) => None,
            LayerProof::Convolution(..) => None,
            LayerProof::Reshape => None,
            LayerProof::Activation(ActivationProof { lookup, .. })
            | LayerProof::Requant(RequantProof { lookup, .. })
            | LayerProof::Pooling(PoolingProof { lookup, .. }) => Some(lookup.fractional_outputs()),
        }
    }
}
impl<T: Number> std::fmt::Display for Layer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe())
    }
}
