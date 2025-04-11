use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    commit::precommit::PolyID, iop::context::ContextAux, layers::LayerCtx, tensor::Number, Element, ScalingFactor, Tensor
};

use super::{common::{Op, ProvableOp, QuantizableOp}, Layer};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Reshape;

impl<N: Number> Op<N> for Reshape {
    fn output_shape(&self) -> Vec<usize> {
        vec![1]
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
        id: PolyID,
        aux: ContextAux,
    ) -> Option<(LayerCtx<E>, ContextAux)>
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        None
    }
}
