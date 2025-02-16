use ff_ext::ExtensionField;
use anyhow::Context as CC;
use itertools::Itertools;
use multilinear_extensions::virtual_poly::VPAuxInfo;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use transcript::Transcript;
use crate::activation::ActivationCtx;
use crate::iop::precommit::PolyID;
use crate::iop::precommit;
use crate::model::{Layer, Model};

/// Common information between prover and verifier
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Context<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Dimensions of the polynomials necessary to verify the sumcheck proofs
    /// These poly are from the matrices weights
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub polys_aux: Vec<(PolyID, VPAuxInfo<E>)>,
    /// Context related to the commitment and accumulation of claims related to the weights of model.
    /// This part contains the commitment of the weights.
    pub weights: precommit::Context<E>,

    /// Context holding the lookup tables for activation, e.g. the MLEs of the input and output columns for 
    /// RELU for example
    pub activation: ActivationCtx<E>,
}

impl<E: ExtensionField> Context<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Generates a context to give to the verifier that contains informations about the polynomials
    /// to prove at each step.
    /// INFO: it _assumes_ the model is already well padded to power of twos.
    pub fn generate(model: &Model) -> anyhow::Result<Self> {
        let auxs = model
            .layers()
            .map(|(id, layer)| {
                match layer {
                    Layer::Dense(matrix) => {
                        // construct dimension of the polynomial given to the sumcheck
                        let ncols = matrix.ncols();
                        // each poly is only two polynomial right now: matrix and vector
                        // for matrix, each time we fix the variables related to rows so we are only left
                        // with the variables related to columns
                        let matrix_num_vars = ncols.ilog2() as usize;
                        let vector_num_vars = matrix_num_vars;
                        // there is only one product (i.e. quadratic sumcheck)
                        (
                            id,
                            VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                                matrix_num_vars,
                                vector_num_vars,
                            ]]),
                        )
                    } 
                    _ => unimplemented!(),
                }
            })
            .rev()
            .collect_vec();
        let commit_ctx = precommit::Context::generate_from_model(model)
            .context("can't generate context for commitment part")?;
        let activation = ActivationCtx::new();
        Ok(Self {
            polys_aux: auxs,
            weights: commit_ctx,
            activation,
        })
    }

    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        for (id, poly_info) in self.polys_aux.iter() {
            t.append_field_element(&E::BaseField::from(*id as u64));
            poly_info.write_to_transcript(t);
        }
        self.weights.write_to_transcript(t)?;
        Ok(())
    }
}