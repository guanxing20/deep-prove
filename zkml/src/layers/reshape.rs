use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

use crate::{
    commit::precommit::PolyID, iop::context::ContextAux, layers::LayerCtx, padding::PaddingMode, tensor::Number, Element, NextPowerOfTwo, ScalingFactor, Tensor
};

use super::{
    provable::{LayerOut, Op, OpInfo, ProvableOp, ProveInfo}, Layer
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reshape;

impl OpInfo for Reshape {
    fn output_shapes(&self, input_shapes: &[Vec<usize>], _padding_mode: PaddingMode) -> Vec<Vec<usize>> {
        input_shapes.into_iter().map(|s|
            vec![s.iter().product()]
        ).collect()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Reshape".to_string()
    }
}

impl<N: Number, E: ExtensionField> Op<N, E> for Reshape {
    fn evaluate(&self, inputs: &[&Tensor<N>], _unpadded_input_shapes: Vec<Vec<usize>>) -> Result<LayerOut<N, E>, super::provable::ProvableOpError> {
        if inputs.len() != 1 {
            return Err(super::provable::ProvableOpError::ParameterError(
                "Reshape expects exactly one input".to_string(),
            ));
        };
        let input = inputs[0];
        Ok(LayerOut::from_vec(vec![input.flatten()]))
    }
}

impl<E> ProveInfo<E> for Reshape 
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(
        &self,
        _id: PolyID,
        mut aux: ContextAux,
    ) -> (LayerCtx<E>, ContextAux) {
        aux.last_output_shape.iter_mut().for_each(|s| 
            *s = s.next_power_of_two()
        );
        (LayerCtx::Reshape, aux)
    }
}

impl<E, T, N> ProvableOp<E, T, N> for Reshape 
where 
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    T: Transcript<E>,
    N: Number,
{
    fn is_provable(&self) -> bool {
        false
    }
}

impl Reshape {
    pub(crate) fn quantize(&self, _s: &ScalingFactor, _bias_s: Option<&ScalingFactor>) -> Layer<Element> {
        Layer::Reshape(Reshape)
    }
}
