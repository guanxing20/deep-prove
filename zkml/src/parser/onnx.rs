use crate::{
    layers::{activation::Activation, convolution::Convolution, provable::{Edge, NodeId, Op as DOP, ProvableModel, ProvableNode, ProvableOp, ProvableOpError}, Layer},
    quantization::Fieldizer,
};
use anyhow::{bail, ensure, Context};
use ff_ext::ExtensionField;
use goldilocks::GoldilocksExt2;
use once_cell::sync::Lazy;
use std::{collections::HashMap, sync::Mutex};
use tract_onnx::{
    prelude::*,
    tract_core::{self, ops::{binary::TypedBinOp, cnn::Conv, einsum::EinSum}},
    tract_hir::{internal::Expansion, ops::{cnn::PaddingSpec, konst::Const, math::Add}},
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
    #[allow(unused_variables)]
    let op_name = &curr_node.name;
    if let Some(node_type) = ALL_NODES_NAMES.iter().find(|&&x| op_name.contains(x)) {
        match *node_type {
            "Conv" => load_conv(model, *curr_node_id, curr_node, iter)?,
            "Gemm.ab" => load_gemm(model, *curr_node_id, curr_node, iter)?,
            "Relu" => load_relu(model, *curr_node_id, curr_node, iter)?,
            _ => panic!("Unknown node type: {}", op_name),
        }
    } else {
        return err(format!("Unknown node type: {}", op_name));
    };
    Ok(())
}

type CustomNode = crate::layers::provable::ProvableNode<f32>;

fn load_relu<'a, I: Iterator<Item = &'a usize>>(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    iter: &mut I,
) -> Result<CustomNode,ProvableOpError> {
    let relu = crate::layers::activation::Relu::new();
    // find the input node that corresponds to the const input of Relu - since tract_onnx transforms
    // a relu operation into Max(input, Const(0))
    // the input node would be the other one.
    ensure_onnx!(node.inputs.len() == 2, "Relu {} must have 2 inputs", node.name);
    let real_input_id= match model.node(node.inputs[1].node).op_as::<Const>() {
        Some(_) => {
            node.inputs[0] 
        }
        None => {
            ensure_onnx!(model.node(node.inputs[0].node).op_as::<Const>().is_some(), "Relu {} has no constant input", node.name);
            node.inputs[1]
        }
    };
    let provable_node = crate::layers::provable::ProvableNode::new(
        vec![Edge::new(real_input_id.node, real_input_id.slot)], 
        Layer::Activation(Activation::Relu(relu)),
    );
    Ok(provable_node)
}

fn load_gemm<'a, I: Iterator<Item = &'a usize>>(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    iter: &mut I,
) -> Result<CustomNode,ProvableOpError> {
    let _matrix = downcast_to::<EinSum>(node)?;
    // TODO: we only support matvec for now for onnx models
    // Fetch the input which is constant (e.g. the weights) 
    ensure_onnx!(node.inputs.len() == 2, "Gemm {} must have 2 inputs", node.name);
    let Some(weight_link) = node.inputs.iter().rev().find(|&x|  is_const(model.node(x.node))) else {
        return err(format!("Gemm {} has no constant input", node.name));
    };
    let weight = extract_const_tensor(model.node(weight_link.node))?;
    // find the input node
    let Some(input_link) = node.inputs.iter().find(|&x| x.node != weight_link.node) else {
        return err(format!("Gemm {} has no input", node.name));
    };

    // also extract the bias if any. If there is one, that means the next node is a Add node and we
    // must make sure one of the inputs is the current matrix node. Otherwise that's just a normal add that we don't support.
    // TODO: support general case with Add layer.
    let mut pit = iter.peekable();
    let next_node_id = pit.peek().ok_or(ProvableOpError::OnnxParsingError(format!("Gemm {} has no next node", node.name)))?;
    let next_node = model.node(**next_node_id);
    let bias_node = match downcast_to::<TypedBinOp>(next_node) {
        _ if next_node.inputs.len() != 2 => {
            // no bias, just return the matrix node
            None
        }
        Ok(binop) if binop.0.is::<tract_core::ops::math::Add>() => {
            match next_node.inputs.iter().enumerate().find(|(_i,&x)| x.node == node_id) {
                Some((idx, ..)) => {
                    // since only two elements, we can just do 1 - idx
                    let bias_input = next_node.inputs[1 - idx];
                    let bias_node = model.node(bias_input.node);
                    // in that case, we move on the iterator, since we already saw the bias node and the Add is part of the dense layer
                    iter.next()
                }
                None => {
                    // no bias, just return the matrix node
                    None
                }
            }
        }
        _ => {
            // no bias, just return the matrix node
            None
        }
    };
    let bias_tensor = match bias_node {
        Some(node_id) => extract_const_tensor(model.node(*node_id))?,
        None => crate::Tensor::zeros(vec![weight.shape[0]]),
    };
    ensure_onnx!(bias_tensor.shape.len() == 1, "Bias tensor must be 1D");
    ensure_onnx!(bias_tensor.shape[0] == weight.shape[0], "Bias tensor must have same size as filter's rows");
    let dense = crate::layers::dense::Dense::new(weight, bias_tensor);
    let provable_node = crate::layers::provable::ProvableNode::new(
        vec![Edge::new(input_link.node, input_link.slot)], 
        Layer::Dense(dense),
    );
    Ok(provable_node)
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
    let filter_node = model.node(filter_link.node);
    let bias_node = model.node(bias_link.node);
    let filter_const = extract_const_tensor(filter_node)?;
    let bias_const = extract_const_tensor(bias_node)?;
    let conv = Convolution::new(filter_const, bias_const);
    let provable_node = crate::layers::provable::ProvableNode::new(
        vec![Edge::new(input_link.node, input_link.slot)], 
        Layer::Convolution(conv),
    );
    Ok(provable_node)
}

fn is_const(node: &OnnxNode) -> bool {
    downcast_to::<Const>(node).is_ok()
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

enum LinearOp {
    MatVec,
}

impl LinearOp {
    fn from_einsum(einsum: &EinSum) -> Result<Self, ProvableOpError> {
        let str_eq = einsum.axes.to_string();
        let io = str_eq.split("->").collect::<Vec<_>>();
        ensure_onnx!(io.len() == 2, "Einsum {:?} must have only 1 operation", einsum);
        // mk,nk
        let inputs= io[0].split(',').collect::<Vec<_>>();
        ensure_onnx!(inputs.len() == 2, "Einsum {:?} must have 2 inputs", einsum);
        // mk,nk
        if inputs[0].len() != 2  || inputs[1].len() != 2 {
            return err(format!("Einsum {:?} must have 2D inputs", einsum));
        }
        
        let first_dim = inputs[0].chars().nth(0).ok_or(ProvableOpError::OnnxParsingError(format!("Einsum {:?} must have 2D inputs", einsum)))?;
        let last_dim = inputs[1].chars().nth(1).ok_or(ProvableOpError::OnnxParsingError(format!("Einsum {:?} must have 2D inputs", einsum)))?;
        ensure_onnx!(io[1].len() == 2, "Einsum {:?} must have 2 outputs dimensions", einsum);
        let mut outputs = io[1].chars();;
        let first_output_dim = outputs.nth(0).ok_or(ProvableOpError::OnnxParsingError(format!("Einsum {:?} must have 2D inputs", einsum)))?;
        let last_output_dim = outputs.nth(1).ok_or(ProvableOpError::OnnxParsingError(format!("Einsum: {:?} must have 2D inputs", einsum)))?;
        ensure_onnx!(first_dim == first_output_dim && last_dim == last_output_dim, "Einsum {:?} must have matching dimensions", einsum);
        Ok(Self::MatVec)
    }
}

fn err<T>(msg: String) -> Result<T, ProvableOpError> {
    Err(ProvableOpError::OnnxParsingError(msg).into())
}

fn downcast_to<T: Op>(node: &OnnxNode) -> Result<&T, ProvableOpError> {
    match node.op_as::<T>() {
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
