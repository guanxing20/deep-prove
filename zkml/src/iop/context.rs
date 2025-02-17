use crate::{
    activation::{Activation, ActivationCtx},
    iop::precommit::{self, PolyID},
    model::{Layer, Model},
};
use anyhow::Context as CC;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::virtual_poly::VPAuxInfo;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

/// Describes a steps wrt the polynomial to be proven/looked at. Verifier needs to know
/// the sequence of steps and the type of each step from the setup phase so it can make sure the prover is not
/// cheating on this.
#[derive(Clone,Debug, Serialize,Deserialize)]
pub enum StepInfo<E> {
    Dense { 
        poly_id: PolyID,
        poly_aux: VPAuxInfo<E>
    },
    Activation {
        poly_id: PolyID,
        poly_aux: VPAuxInfo<E>
    }
}

impl<E> StepInfo<E> {
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense {poly_id: _, poly_aux : _} => "Dense".to_string(),
            Self::Activation { poly_id: _, poly_aux : _} => "Activation".to_string(),
        }
    }
}

/// Common information between prover and verifier
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Context<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub steps_kind: Vec<StepInfo<E>>,
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
                        StepInfo::Dense {
                            poly_id: id, 
                            poly_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                                matrix_num_vars,
                                vector_num_vars,
                            ]]),
                        }
                    }
                    Layer::Activation(_) => {
                        // TODO: make a full list of all polys at each step and refer to them in the prover/verifier part
                        unimplemented!()
                    }
                }
            })
            .rev()
            .collect_vec();
        let commit_ctx = precommit::Context::generate_from_model(model)
            .context("can't generate context for commitment part")?;
        let activation = ActivationCtx::new();
        Ok(Self {
            steps_kind: auxs,
            weights: commit_ctx,
            activation,
        })
    }

    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        for steps in self.steps_kind.iter() {
            match steps {
                StepInfo::Dense { poly_id, poly_aux } => {
                    t.append_field_element(&E::BaseField::from(*poly_id as u64));
                    poly_aux.write_to_transcript(t);
                }
                StepInfo::Activation { poly_id, poly_aux } => {
                    t.append_field_element(&E::BaseField::from(*poly_id as u64));
                    poly_aux.write_to_transcript(t);
                }
            }
        }
        self.weights.write_to_transcript(t)?;
        Ok(())
    }
}
