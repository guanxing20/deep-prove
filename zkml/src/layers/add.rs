use anyhow::{bail, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};

use crate::{
    Element, ScalingFactor, ScalingStrategy, Tensor,
    layers::provable::{Evaluate, NodeId, OpInfo, QuantizeOp, QuantizeOutput},
    padding::PaddingMode,
    tensor::{Number, Shape},
};

use super::provable::LayerOut;

/// Add layer that adds two tensors together.
/// If there is two inputs, no static weight, then the output shape is the same as the first input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Add<N> {
    /// shape is the unpadded shape of the operand
    operand: Option<(Tensor<N>, Shape)>,
}

impl<N: Number> Default for Add<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Number> Add<N> {
    pub fn new() -> Self {
        Self { operand: None }
    }
    pub fn new_with(operand: Tensor<N>, unpadded_shape: Shape) -> Self {
        Self {
            operand: Some((operand, unpadded_shape)),
        }
    }
}

impl<N: Number> Evaluate<N> for Add<N> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        let result = if inputs.len() == 2 {
            ensure!(
                inputs[0].get_shape().product() == inputs[1].get_shape().product(),
                "Add layer expects inputs to have the same shape: {:?} vs {:?}",
                inputs[0].get_shape(),
                inputs[1].get_shape()
            );
            inputs[0].add(inputs[1])
        } else if inputs.len() == 1 {
            ensure!(
                self.operand.is_some(),
                "Add operand can't be None if there is only one input"
            );
            ensure!(
                inputs[0].get_shape().product()
                    == self.operand.as_ref().unwrap().0.get_shape().product(),
                "Add layer expects input and operand to have the same shape: {:?} vs {:?}",
                inputs[0].get_shape(),
                self.operand.as_ref().unwrap().0.get_shape()
            );
            inputs[0].add(&self.operand.as_ref().unwrap().0)
        } else {
            bail!("Add layer expects 1 or 2 inputs, got {}", inputs.len());
        };
        Ok(LayerOut::from_vec(vec![result]))
    }
}

impl<N> OpInfo for Add<N> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        if let Some((_, og_shape)) = &self.operand {
            assert!(
                *og_shape == input_shapes[0],
                "Add layer operand shape mismatch: {:?} vs {:?}",
                og_shape,
                &input_shapes[0]
            );
        }
        match padding_mode {
            PaddingMode::NoPadding => input_shapes.to_vec(),
            PaddingMode::Padding => input_shapes
                .iter()
                .map(|shape| shape.next_power_of_two())
                .collect(),
        }
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        "Add".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl QuantizeOp for Add<f32> {
    type QuantizedOp = Add<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeId,
        _input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::Element;

    use super::*;

    #[test]
    fn test_add() {
        let add = Add::new();
        let t1 = Tensor::<Element>::random(&vec![2, 2].into());
        let t2 = Tensor::<Element>::random(&vec![2, 2].into());
        let result = add
            .evaluate::<GoldilocksExt2>(&[&t1, &t2], vec![vec![2, 2].into(), vec![2, 2].into()])
            .unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    result.outputs[0].get(vec![i, j]),
                    t1.get(vec![i, j]) + t2.get(vec![i, j])
                );
            }
        }
        let add = Add::new_with(t1.clone(), t1.get_shape().into());
        let result = add
            .evaluate::<GoldilocksExt2>(&[&t2], vec![vec![2, 2].into()])
            .unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    result.outputs[0].get(vec![i, j]),
                    t1.get(vec![i, j]) + t2.get(vec![i, j])
                );
            }
        }
    }
}
