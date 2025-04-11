use ff_ext::ExtensionField;
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    Element, ScalingFactor, Tensor, commit::precommit::PolyID, iop::context::ContextAux,
    tensor::Number,
};

use super::{Layer, LayerCtx};

pub trait Op<N: Number>:
    Clone + std::fmt::Debug + Sync + Send + Serialize + DeserializeOwned
{
    fn describe(&self) -> String;
    fn output_shape(&self) -> Vec<usize>;
    fn op(&self, input: &Tensor<N>) -> Tensor<N>;
}

pub trait QuantizableOp: Op<f32> {
    fn quantize(&self, s: &ScalingFactor, bias_s: Option<&ScalingFactor>) -> Layer<Element>;
}

pub trait ProvableOp: Op<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: PolyID,
        aux: ContextAux,
    ) -> Option<(LayerCtx<E>, ContextAux)>
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned;
}
