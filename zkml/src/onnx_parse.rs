use anyhow::{Error, Result, bail, ensure};
use itertools::Itertools;
use std::{collections::HashMap, i8, path::Path};
use tract_onnx::{pb::NodeProto, prelude::*};

use crate::{
    matrix::Matrix, model::{Layer, Model}, quantization::{QuantInteger, Quantizer}, Element
};

#[derive(Debug, Clone)]
struct Gemm {
    name: String,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

// Given a ONNX node, build a struct which contains information about the Gemm
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

// Given serialized data and its tract DatumType, build a tract tensor.
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

// Given a flat vector, converts it to matrix in row major form
fn reshape<T: Clone>(flat_vec: Vec<T>, rows: usize, cols: usize) -> Option<Vec<Vec<T>>> {
    if flat_vec.len() != rows * cols {
        return None; // Return None if dimensions don't match the number of elements
    }

    Some(flat_vec.chunks(cols).map(|chunk| chunk.to_vec()).collect())
}

fn concat_column(matrix: Vec<Vec<Element>>, column: Vec<Vec<Element>>) -> Result<Vec<Vec<Element>>> {
    if matrix.len() != column.len() {
        bail!("Column length must match matrix row count");
    }

    let new_matrix: Vec<Vec<Element>> = matrix
        .into_iter()
        .zip(column.into_iter())
        .map(|(mut row, col_val)| {
            if col_val.len() != 1 {
                bail!("Column vector must have a single value per row");
            }
            row.push(col_val[0]); // Append the single column value
            Ok(row)
        })
        .collect::<Result<Vec<_>>>()?; // Collect results into Vec<Vec<u64>>

    Ok(new_matrix)
}

fn fetch_weight_bias_as_mat<Q: Quantizer<Element>>(
    weight_or_bias: &str,
    node: &NodeProto,
    initializers: &HashMap<String, Tensor>,
) -> Result<Vec<Vec<Element>>> {
    ensure!(weight_or_bias == "weight" || weight_or_bias == "bias");

    let alpha_or_beta = node
        .attribute
        .iter()
        .filter(|x| {
            x.name.contains(match weight_or_bias {
                "weight" => "alpha",
                _ => "beta",
            })
        })
        .map(|x| x.f)
        .collect_vec();
    let alpha_or_beta = alpha_or_beta[0];

    let tensor_vec = node
        .input
        .iter()
        .filter(|x| x.contains(weight_or_bias))
        .filter_map(|key| initializers.get(key).cloned())
        .collect_vec();

    // If a node is Gemm, then it has only one tensor of the form "fcN.weight"
    let tensor_t = tensor_vec[0].clone();
    let tensor_t_f32 = tensor_t.as_slice::<f32>().unwrap().to_vec();
    let tensor_t_f32 = tensor_t_f32.iter().map(|x| x * alpha_or_beta).collect_vec();
    let tensor_f = tensor_t_f32.iter().map(Q::from_f32_unsafe).collect_vec();

    let (rows, cols) = match tensor_t.shape().len() {
        1 => (tensor_t.shape()[0], 1),
        2 => (tensor_t.shape()[0], tensor_t.shape()[1]),
        _ => bail!("Invalid tensor shape: expected 1D or 2D tensor"),
    };

    let field_matrix = reshape(tensor_f, rows, cols).unwrap();

    Ok(field_matrix)
}

pub fn load_mlp<Q: Quantizer<Element>>(filepath: &str) -> Result<Model> {
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
                let matrix_weight = fetch_weight_bias_as_mat::<Q>("weight", node, &initializers)?;
                let matrix_bias = fetch_weight_bias_as_mat::<Q>("bias", node, &initializers)?;

                let matrix = concat_column(matrix_weight, matrix_bias)?;
                let matrix = Matrix::<Element>::from_coeffs(matrix)
                    .unwrap()
                    .pad_next_power_of_two();
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
        let input = random_vector::<Element>(model.input_shape()[0]);

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
