use std::marker::PhantomData;

use ark_std::rand::thread_rng;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
use serde::{Deserialize, Serialize};
use transcript::Transcript;

use crate::{testing::random_field_vector, Claim};

type MLE<E> = DenseMultilinearExtension<E>;
#[derive(Clone,Serialize,Deserialize)]
pub struct Proof<E> {
    // one commitment per "column" in the lookups
    //lookups_C: Vec<(Commitment<E>,Opening,Claim)>,
    //gkr_proof: GKRProof<E>,
    //multi_C: Commitment<E>,
    // One per columns in the lookup table (so one for input and one for output at least)
    pub claims: Vec<Claim<E>>,
    _p: PhantomData<E>,
}

struct Context<E> {
    _p: PhantomData<E>,
}

pub trait LookupProtocol<E: ExtensionField> {
    // lookups has dimension (N,N) in case of two columns
    // with N lookup "op".
    // we must pad lookups MLEs to same dim than table
    // table can come from context
    // e.g table[i].num_vars() == lookups[i].num_vars()
    fn prove<T: Transcript<E>>(table: Vec<MLE<E>>, lookups: Vec<MLE<E>>,t: &mut T) -> anyhow::Result<Proof<E>>;
    
    // commitments to the lookups, one commitment per "column"
    fn verify<T: Transcript<E>>(ctx: Context<E>, proof: Proof<E>, t: &mut T) -> anyhow::Result<()>;
}

pub struct DummyLookup {}

impl<E: ExtensionField> LookupProtocol<E> for DummyLookup {
    fn prove<T: Transcript<E>>(table: Vec<MLE<E>>, lookups: Vec<MLE<E>>,t: &mut T) -> anyhow::Result<Proof<E>> {
        assert_eq!(table.len(),lookups.len());
        assert!(table.iter().zip(lookups.iter()).all(|(t,l)| t.num_vars() == l.num_vars()));
        let claims = lookups.iter().map(|l| {
            // TODO: to replace via proper lookup protocol output
            let point = random_field_vector(l.num_vars());
            let eval = l.evaluate(&point);
            Claim {
                point,
                eval,
            }
        }).collect_vec();
        Ok(Proof {
            claims,
            _p: PhantomData,
        })
    }

    fn verify<T: Transcript<E>>(ctx: Context<E>, proof: Proof<E>, t: &mut T) -> anyhow::Result<()> {
        todo!()
    }
}