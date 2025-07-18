use serde::{Deserialize, Serialize};

use super::{Element, Proof as ProofG, model::Model, quantization::ModelMetadata};

pub mod v1;

/// A versioned enum representing a deep prove proving request
#[derive(Serialize, Deserialize)]
pub enum DeepProveRequest {
    /// Version 1
    V1(v1::DeepProveRequest),
}

/// A versioned enum representing a deep prove proving response
#[derive(Serialize, Deserialize)]
pub enum DeepProveResponse {
    /// Version 1
    V1(v1::DeepProveResponse),
}
