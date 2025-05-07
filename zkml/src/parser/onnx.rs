use crate::{
    layers::{convolution::Convolution, provable::{Edge, Op as DOP, ProvableModel, ProvableNode, ProvableOp, ProvableOpError}},
    quantization::Fieldizer,
};
use anyhow::{bail, ensure};
use ff_ext::ExtensionField;
use goldilocks::GoldilocksExt2;
use once_cell::sync::Lazy;
use std::{collections::HashMap, sync::Mutex};
use tract_onnx::{
    prelude::*,
    tract_core::ops::{cnn::Conv, einsum::EinSum},
    tract_hir::{internal::Expansion, ops::{cnn::PaddingSpec, konst::Const}},
};
use transcript::Transcript;

type OnnxModel = Graph<TypedFact, Box<dyn TypedOp + 'static>>;
type OnnxNode = Node<TypedFact, Box<dyn TypedOp + 'static>>;

macro_rules! ensure_onnx {
        // Match with format args
        ($cond:expr, $err_fmt:literal, $($args:expr),+ $(,)?) => {
            if !$cond {
                return Err(ProvableOpError::OnnxParsingError(format!($err_fmt, $($args),+)).into());
            }
        };
        // Match with plain string (no args)
        ($cond:expr, $err_msg:literal $(,)?) => {
            if !$cond {
                return Err(ProvableOpError::OnnxParsingError($err_msg.to_string()).into());
            }
        };
    }

pub fn from_path(path: &str) -> anyhow::Result<()> {
    let model = tract_onnx::onnx().model_for_path(path)?.into_typed()?;
    let plan = SimplePlan::new(model)?;
    let model = plan.model();
    let inference_order = plan.order_without_consts();
    let input_node = model.node(inference_order[0]);
    let mut it = inference_order[1..].iter();
    while !it.is_empty() {
        parse_node(model, &mut it)?;
    }
    unimplemented!()
}

const ALL_NODES_NAMES: &[&str] = &["Conv", "MatMul"];

fn parse_node<'a, I: Iterator<Item = &'a usize>>(
    model: &OnnxModel,
    iter: &mut I,
) -> Result<(),ProvableOpError> {
    let curr_node_id = iter.next().ok_or(anyhow::anyhow!("No nodes left"))?;
    let curr_node = model.node(*curr_node_id);
    let op_name = curr_node.op().name();
    if let Some(node_type) = ALL_NODES_NAMES.iter().find(|&&x| op_name.contains(x)) {
        match *node_type {
            "Conv" => load_conv(model, curr_node_id, curr_node, iter)?,
            //"MatMul" => load_gemm(model, curr_node, iter)?,
            _ => panic!("Unknown node type: {}", op_name),
        }
    } else {
        return err(format!("Unknown node type: {}", op_name));
    };
    Ok(())
}

type CustomNode = crate::layers::provable::Node<Box<dyn DOP<f32, GoldilocksExt2>>>;

fn load_gemm<'a, I: Iterator<Item = &'a usize>>(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    iter: &mut I,
) -> Result<CustomNode,ProvableOpError> {
    let matrix = downcast_to::<EinSum>(node)?;
    // also extract the bias if any. If there is one, that means the next node is a Add node and we
    // must make sure one of the inputs is the current matrix node. Otherwise that's just a normal add that we don't support.
    // TODO: support general case with Add layer.
    let next_node_id = iter.peekable().peek().ok_or(ProvableOpError::OnnxParsingError(format!("Gemm {} has no next node", node.name)))?;
    let next_node = model.node(**next_node_id);
    let bias = match downcast_to::<Add>(&next_node) {
        Ok(add_node) => {
            assert!(add_node.inputs.len() == 2, "Add node must have 2 inputs");
            match add_node.inputs.iter().enumerate().find(|(i,&x)| x.node == node_id) {
                Some((idx, ..)) => {
                    let bias_input = add_node.inputs[1 - idx];
                    
                }

                Some((1, ..)) => {
                    // bias is the current node
                }
                None => {
                    // no bias, just return the matrix node
                }
            }
        }
        Err(_) => {
            // no bias, just return the matrix node
        }
    };
    unimplemented!()
}

fn load_conv<'a, I: Iterator<Item = &'a usize>>(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    iter: &mut I,
) -> Result<CustomNode,ProvableOpError> {
    let conv_node = downcast_to::<Conv>(node)?;
    // TODO: once we support different padding and strides, extract the data in this function
    check_conv2d_attributes(conv_node)?;
    // TODO: support for conv without bias
    ensure_onnx!(node.inputs.len() == 3, "ONNX Conv {} must have at least 3 inputs: {}", node.name, node.inputs.len());
    let input_link = node.inputs[0];
    let filter_link = node.inputs[1];
    let bias_link = node.inputs[2];
    let input_node = model.node(input_link.node);
    let filter_node = model.node(filter_link.node);
    let bias_node = model.node(bias_link.node);
    let filter_const = extract_const_tensor(filter_node)?;
    let bias_const = extract_const_tensor(bias_node)?;
    let conv = Convolution::new_raw(filter_const, bias_const);
    let provable_node = crate::layers::provable::Node::new_raw(vec![Edge::new(input_link.node, 0)], Box::new(conv));
    Ok(provable_node)
}

fn extract_const_tensor(node: &OnnxNode) -> Result<crate::Tensor<f32>, ProvableOpError> {
    let tensor = downcast_to::<Const>(node)?;
    let slice = tensor.0.as_slice::<f32>()?;
    ensure_onnx!(node.outputs.len() == 1, "constant output shape len == 1");
    let Some(shape ) = node.outputs[0].fact.shape.as_concrete() else {
        return err(format!("Filter shape {} is not concrete", node.name));
    };
    Ok(crate::Tensor::new(shape.to_vec(), slice.to_vec()))
}

/// Get the conv2d attributes and assert if supported by DeepProve
fn check_conv2d_attributes(node: &Conv) -> Result<(), ProvableOpError> {
    let Some(ref strides) = node.pool_spec.strides else {
        return err(format!("Conv has no strides: {}", node.name()));
    };
    ensure_onnx!(strides.iter().all(|&x| x == 1), "Strides must be {}", 1);
    ensure_onnx!(strides.iter().all(|&x| x == 1), "Strides must be {}", 1);
    let PaddingSpec::Explicit(ref pad0, ref pad1) = &node.pool_spec.padding else {
        return err(format!("Conv has no pads: {}", node.name()));
    };
    ensure_onnx!(
        pad0.iter().all(|&x| x == 0),
        "Padding for {}must be 0s: {:?}",
        node.name(),
        pad0,
    );
    ensure_onnx!(
        pad1.iter().all(|&x| x == 0),
        "Padding for {}must be 0s: {:?}",
        node.name(),
        pad1,
    );
    let Some(ref dilations) = node.pool_spec.dilations else {
        return Err(ProvableOpError::OnnxParsingError(format!(
            "Conv has no dilations: {}",
            node.name()
        ))
        .into());
    };
    ensure_onnx!(
        dilations.iter().all(|&x| x == 1),
        "Dilations for {} must be 1: {:?}",
        node.name(),
        dilations
    );
    let kernel_shape = &node.pool_spec.kernel_shape;
    ensure_onnx!(
        kernel_shape.iter().all(|&x| x > 1),
        "Kernel shape for {} must be > 1: {:?}",
        node.name(),
        kernel_shape
    );
    ensure_onnx!(kernel_shape.len() == 2, "Kernel shape for {} must be 2D: {:?}", node.name(), kernel_shape);
    ensure_onnx!(kernel_shape[0] == kernel_shape[1], "Kernel shape for {} must be square: {:?}", node.name(), kernel_shape);
    Ok(())
}

fn err<T>(msg: String) -> Result<T, ProvableOpError> {
    Err(ProvableOpError::OnnxParsingError(msg).into())
}

fn downcast_to<T>(node: &OnnxNode) -> Result<&T, ProvableOpError> {
    match node.op().downcast_ref::<T>() {
        Some(b) => Ok(b),
        None => {
            return Err(ProvableOpError::OnnxParsingError(format!(
                "Node {} is not a {}",
                node.name,
                std::any::type_name::<T>()
            ))
            .into());
        }   
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_load_conv() {
        let model = from_path("bench-cnn/model.onnx").unwrap();
    }
}
