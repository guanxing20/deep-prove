use anyhow::{Error, Result};
use itertools::Itertools;
use std::{collections::HashMap, path::Path};
use tract_onnx::{pb::NodeProto, prelude::*};

use crate::{
    matrix::Matrix,
    model::{Layer, Model}, Element,
};

#[derive(Debug, Clone)]
struct Gemm {
    name: String,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

fn build_gemm(node: &NodeProto) -> Result<Gemm> {
    let name = node.name.to_string();
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.);
    let beta = node.get_attr_opt("beta")?.unwrap_or(1.);
    let trans_a = node.get_attr_opt("transA")?.unwrap_or(false);
    let trans_b = node.get_attr_opt("transB")?.unwrap_or(false);
    let gemm = Gemm {
        name,
        alpha,
        beta,
        trans_a,
        trans_b,
    };
    Ok(gemm)
}




fn create_tensor(shape: Vec<usize>, dt: DatumType, data: &[u8]) -> TractResult<Tensor> {
    unsafe {
        match dt {
            DatumType::U8 => Tensor::from_raw::<u8>(&shape, data),
            DatumType::U16 => Tensor::from_raw::<u16>(&shape, data),
            DatumType::U32 => Tensor::from_raw::<u32>(&shape, data),
            DatumType::U64 => Tensor::from_raw::<u64>(&shape, data),
            DatumType::I8 => Tensor::from_raw::<i8>(&shape, data),
            DatumType::I16 => Tensor::from_raw::<i16>(&shape, data),
            DatumType::I32 => Tensor::from_raw::<i32>(&shape, data),
            DatumType::I64 => Tensor::from_raw::<i64>(&shape, data),
            DatumType::F16 => Tensor::from_raw::<f16>(&shape, data),
            DatumType::F32 => Tensor::from_raw::<f32>(&shape, data),
            DatumType::F64 => Tensor::from_raw::<f64>(&shape, data),
            DatumType::Bool => Ok(Tensor::from_raw::<u8>(&shape, data)?
                .into_array::<u8>()?
                .mapv(|x| x != 0)
                .into()),
            _ => unimplemented!("create_tensor: Failed"),
        }
    }
}

fn is_mlp(filepath: &str) -> Result<bool> {
    let activation_functions = ["Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Selu"];

    let mut is_mlp = true;
    let mut prev_was_gemm_or_matmul = false;

    let model = tract_onnx::onnx()
        .proto_model_for_path(filepath)
        .map_err(|e| Error::msg(format!("Failed to load model: {:?}", e)))?;
    let graph = model.graph.unwrap();

    for node in graph.node.iter() {
        if node.op_type == "Gemm" || node.op_type == "MatMul" {
            if prev_was_gemm_or_matmul {
                is_mlp = false;
                break;
            }
            prev_was_gemm_or_matmul = true;
        } else if activation_functions.contains(&node.op_type.as_str()) {
            if !prev_was_gemm_or_matmul {
                is_mlp = false;
                break;
            }
            prev_was_gemm_or_matmul = false;
        } else {
            is_mlp = false;
            break;
        }
    }

    Ok(is_mlp)
}

fn reshape<T: Clone>(flat_vec: Vec<T>, rows: usize, cols: usize) -> Option<Vec<Vec<T>>> {
    if flat_vec.len() != rows * cols {
        return None; // Return None if dimensions don't match the number of elements
    }

    Some(flat_vec.chunks(cols).map(|chunk| chunk.to_vec()).collect())
}

pub fn load_mlp<Q: Quantizer<Element>>(filepath: &str) -> Result<Model>
{
    if !Path::new(filepath).exists() {
        return Err(Error::msg(format!("File '{}' does not exist", filepath)));
    }
    let result = is_mlp(filepath)?;
    assert!(result == true, "is_mlp: Failed");

    let model = tract_onnx::onnx()
        .proto_model_for_path(filepath)
        .map_err(|e| Error::msg(format!("Failed to load model: {:?}", e)))?;

    let graph = model.graph.unwrap();

    let mut initializers: HashMap<String, Tensor> = HashMap::new();
    for item in graph.initializer {
        let dt = tract_onnx::pb::tensor_proto::DataType::from_i32(item.data_type)
            .unwrap()
            .try_into()?;
        let shape: Vec<usize> = item.dims.iter().map(|&i| i as usize).collect();
        let value = create_tensor(shape, dt, &item.raw_data).unwrap();
        let key = item.name.to_string();
        initializers.insert(key, value);
    }

    let mut layers: Vec<Layer> = Vec::new();
    for node in graph.node.iter() {
        match node.op_type.as_str() {
            "Gemm" => {
                let values = node
                    .input
                    .iter()
                    .filter(|x| x.contains("weight"))
                    .filter_map(|key| initializers.get(key).cloned())
                    .collect_vec();

                let tensor = values[0].clone();
                let tensor_f32 = tensor.as_slice::<f32>().unwrap().to_vec();
                let tensor_f = tensor_f32
                    .iter()
                    .map(Q::from_f32_unsafe)
                    .collect_vec();
                let matrix = reshape(tensor_f, tensor.shape()[0], tensor.shape()[1]).unwrap();
                let matrix = Matrix::<Element>::from_coeffs(matrix).unwrap().pad_next_power_of_two();
                // let matrix = matrix.transpose();
                layers.push(Layer::Dense(matrix));
            }
            _ => (),
        };
    }
    let mut sumcheck_model = Model::new();
    for layer in layers {
        sumcheck_model.add_layer(layer); // Insert each layer
    }

    Ok(sumcheck_model)
}

trait Quantizer<Output> {
    fn from_f32_unsafe(e: &f32) -> Output;
}

impl Quantizer<Element> for Element {
    fn from_f32_unsafe(e: &f32) -> Self {
        let max_u8 = 255u32;
        let scale = 2.0 / max_u8 as f64;
        let zero_point = (max_u8 as f64) / 2.0;

        let scaled = (*e as f64 / scale + zero_point).round() as u32;
        scaled as Element
    }
}

#[cfg(test)]
mod tests {

    use crate::testing::random_vector;

    use super::*;

    use goldilocks::GoldilocksExt2;

    // cargo test --release --package zkml -- onnx_parse::tests::test_tract --nocapture


    type F = GoldilocksExt2;

    #[test]
    fn test_tract() {
        let filepath = "assets/model.onnx";
        let result = load_mlp::<Element>(&filepath);

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_model_run() {
        let filepath = "assets/model.onnx";

        let model = load_mlp::<Element>(&filepath).unwrap();
        let input = random_vector(4);

        let trace = model.run::<F>(input.clone());
        println!("Result: {:?}", trace.final_output());
    }

    #[test]
    fn test_quantize() {
        let input = [0.09039914, -0.07716653];

        println!(
            "Result: {} => {:?}",
            input[0],
            <Element as Quantizer<Element>>::from_f32_unsafe(&input[0])
        );
        println!(
            "Result: {} => {:?}",
            input[1],
            <Element as Quantizer<Element>>::from_f32_unsafe(&input[1])
        );
        println!(
            "Result: {} => {:?}",
            0,
            <Element as Quantizer<Element>>::from_f32_unsafe(&0.0)
        );
        println!(
            "Result: {} => {:?}",
            -1.0,
            <Element as Quantizer<Element>>::from_f32_unsafe(&-1.0)
        );
        println!(
            "Result: {} => {:?}",
            1.0,
            <Element as Quantizer<Element>>::from_f32_unsafe(&1.0)
        );
    }
}
