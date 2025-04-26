use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    Element, NextPowerOfTwo, ScalingFactor, Tensor, commit::precommit::PolyID,
    iop::context::ContextAux, layers::LayerCtx, tensor::Number,
};

use super::{
    Layer,
    common::{Op, ProvableOp, QuantizableOp},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reshape;

impl<N: Number> Op<N> for Reshape {
    fn output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        vec![input_shape.iter().product::<usize>()]
    }

    fn op(&self, input: &Tensor<N>) -> Tensor<N> {
        input.flatten()
    }
    fn describe(&self) -> String {
        "Reshape".to_string()
    }
}

impl QuantizableOp for Reshape {
    fn quantize(&self, _s: &ScalingFactor, _bias_s: Option<&ScalingFactor>) -> Layer<Element> {
        Layer::Reshape(Reshape)
    }
}

impl ProvableOp for Reshape {
    fn step_info<E: ExtensionField>(
        &self,
        _id: PolyID,
        mut aux: ContextAux,
    ) -> (LayerCtx<E>, ContextAux)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        aux.last_output_shape = aux.last_output_shape.next_power_of_two();
        (LayerCtx::Reshape, aux)
    }
}
