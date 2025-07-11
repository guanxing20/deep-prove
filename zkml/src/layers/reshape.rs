use std::ops::{Range, RangeBounds};

use crate::{
    layers::provable::{QuantizeOp, QuantizeOutput},
    padding::PaddingMode,
    tensor::Shape,
};
use anyhow::ensure;
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};

use crate::{Tensor, tensor::Number};

use super::provable::{Evaluate, LayerOut, OpInfo};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Reshape {
    Full(Vec<Shape>),
    // (v1,v2) where
    // - v1 are the indices in the shape of the tensor that we want to remove
    // - v2 are the indices that we add in place
    // e.g. if tensor is [a,b,c], and we give Subspace(1..=2,vec![b/6,c,6]) then the
    // output shape is [a,b/6,c,6]
    Subspace((Range<usize>, Vec<usize>)),
    /// Adds a 1 at the given index in the shape.
    Squeeze(usize),
}

impl Reshape {
    pub fn new_fixed(new_dim: Vec<Shape>) -> Self {
        Self::Full(new_dim)
    }
    pub fn new_subspace<R: RangeBounds<usize>>(to_remove: R, to_add: Vec<usize>) -> Self {
        let start = range_start(&to_remove).expect("invalid start bound");
        let end = range_end(&to_remove).expect("invalid end bound");
        Self::Subspace((Range { start, end }, to_add))
    }
    pub fn new_squeeze(index: usize) -> Self {
        Self::Squeeze(index)
    }
    fn internal_output(&self, input_shapes: &[Shape]) -> anyhow::Result<Vec<Shape>> {
        let new_dims = match self {
            Reshape::Squeeze(index) => {
                ensure!(*index < input_shapes[0].len(), "index out of bounds");
                let mut new_dim = input_shapes[0].clone().into_vec();
                new_dim.insert(*index, 1);
                vec![Shape::new(new_dim)]
            }
            Reshape::Full(ref new_dim) => new_dim.clone(),
            Reshape::Subspace((to_remove, to_add)) => input_shapes
                .iter()
                .map(|shape| {
                    let mut new_shape = shape.clone();
                    new_shape.splice(to_remove.clone(), to_add.clone());
                    new_shape
                })
                .collect::<Vec<Shape>>(),
        };
        ensure!(
            new_dims.len() == input_shapes.len(),
            "new_dims.len() == input_shapes.len()"
        );
        ensure!(
            new_dims
                .iter()
                .zip(input_shapes.iter())
                .all(|(new_dim, input_shape)| new_dim.product() == input_shape.product())
        );
        Ok(new_dims)
    }
}

impl OpInfo for Reshape {
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        match self.internal_output(input_shapes) {
            Ok(out) => out,
            Err(e) => panic!("invalid reshape parameters: {e:?}"),
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        match self {
            Reshape::Squeeze(index) => format!("Reshape: squeeze({index})"),
            Reshape::Full(ref new_dim) => format!("Reshape: fixed {new_dim:?}"),
            Reshape::Subspace(_) => "Reshape: dynamic".to_string(),
        }
    }

    fn is_provable(&self) -> bool {
        false
    }
}

impl<N: Number> Evaluate<N> for Reshape {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        let output_shapes =
            self.internal_output(&inputs.iter().map(|x| x.get_shape()).collect::<Vec<_>>())?;
        #[allow(suspicious_double_ref_op)]
        let out_tensors = inputs.iter().map(|x| x.clone().clone()).collect::<Vec<_>>();
        let out_tensors = output_shapes
            .into_iter()
            .zip(out_tensors)
            .map(|(new_dim, input_tensor)| input_tensor.reshape(new_dim))
            .collect();
        Ok(LayerOut::from_vec(out_tensors))
    }
}

impl QuantizeOp for Reshape {
    type QuantizedOp = Reshape;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: super::provable::NodeId,
        input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        Ok(QuantizeOutput::new(self, input_scaling.to_vec()))
    }
}

fn range_start<R: RangeBounds<usize>>(range: &R) -> Option<usize> {
    match range.start_bound() {
        std::ops::Bound::Included(&s) => Some(s),
        std::ops::Bound::Excluded(&s) => Some(s + 1),
        std::ops::Bound::Unbounded => None,
    }
}

fn range_end<R: RangeBounds<usize>>(range: &R) -> Option<usize> {
    match range.end_bound() {
        std::ops::Bound::Included(&e) => Some(e + 1),
        std::ops::Bound::Excluded(&e) => Some(e),
        std::ops::Bound::Unbounded => None,
    }
}

#[cfg(test)]
mod tests {
    use ff_ext::GoldilocksExt2;

    use crate::Element;

    use super::*;

    #[test]
    fn test_reshape_fixed() {
        let input = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
        );
        let reshape = Reshape::new_fixed(vec![vec![3, 2, 3].into()]);
        let output = reshape
            .evaluate::<GoldilocksExt2>(&[&input], vec![])
            .expect("reshape shouldn't fail");
        assert_eq!(output.outputs[0].get_shape(), vec![3, 2, 3].into());
        assert_eq!(output.outputs[0].get_data(), input.get_data());
    }

    #[test]
    fn test_reshape_squeeze() {
        let input = Tensor::<Element>::new(vec![2, 3].into(), vec![0, 1, 2, 3, 4, 5]);
        let reshape = Reshape::new_squeeze(1);
        let output = reshape
            .evaluate::<GoldilocksExt2>(&[&input], vec![])
            .expect("reshape shouldn't fail");
        assert_eq!(output.outputs[0].get_shape(), vec![2, 1, 3].into());
        assert_eq!(output.outputs[0].get_data(), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_reshape_subspace() {
        let input = Tensor::<Element>::new(
            vec![2, 12].into(),
            (0..24).map(|i| i as Element).collect::<Vec<_>>(),
        );
        println!(
            "expected output: {:?}",
            input
                .get_shape()
                .clone()
                .splice(1..2, vec![3, 4])
                .collect::<Vec<_>>()
        );
        let reshape = Reshape::new_subspace(1..2, vec![3, 4]);
        let output = reshape
            .evaluate::<GoldilocksExt2>(&[&input], vec![])
            .expect("reshape shouldn't fail");
        assert_eq!(output.outputs[0].get_shape(), vec![2, 3, 4].into());
        assert_eq!(
            output.outputs[0].get_data(),
            (0..24).map(|i| i as Element).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_reshape_subspace_range_inclusive() {
        let input = Tensor::<Element>::new(
            vec![2, 6, 2].into(),
            (0..24).map(|i| i as Element).collect::<Vec<_>>(),
        );
        println!(
            "expected output: {:?}",
            input
                .get_shape()
                .clone()
                .splice(1..2, vec![3, 4])
                .collect::<Vec<_>>()
        );
        let reshape = Reshape::new_subspace(1..=2, vec![3, 4]);
        let output = reshape
            .evaluate::<GoldilocksExt2>(&[&input], vec![])
            .expect("reshape shouldn't fail");
        assert_eq!(output.outputs[0].get_shape(), vec![2, 3, 4].into());
        assert_eq!(
            output.outputs[0].get_data(),
            (0..24).map(|i| i as Element).collect::<Vec<_>>()
        );
    }
}
