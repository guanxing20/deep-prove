use ff_ext::ExtensionField;
use serde::{Serialize, de::DeserializeOwned};

use crate::{Element, Tensor, commit::precommit::PolyID, iop::context::ContextAux, tensor::Number};

use super::{Layer, LayerCtx};
