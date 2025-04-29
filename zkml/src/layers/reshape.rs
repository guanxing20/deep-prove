use serde::{Deserialize, Serialize};

use crate::{Tensor, tensor::Number};

use super::common::Op;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reshape;

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
