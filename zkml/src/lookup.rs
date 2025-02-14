use std::marker::PhantomData;

use anyhow::anyhow;
use ff_ext::ExtensionField;
use gkr::structs::{Circuit, IOPProof};
use goldilocks::GoldilocksExt2;
use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
use serde::{Deserialize, Serialize};
use transcript::Transcript;

use crate::{
    Claim,
    logup::{logup_circuit, prove_logup, verify_logup},
    testing::random_field_vector,
};

type MLE<E> = DenseMultilinearExtension<E>;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof<E: ExtensionField> {
    // one commitment per "column" in the lookups
    // lookups_C: Vec<(Commitment<E>,Opening,Claim)>,
    gkr_proof: IOPProof<E>,
    output_claim: E,
    // One per columns in the lookup table (so one for input and one for output at least)
    pub claims: Vec<Claim<E>>,
    _p: PhantomData<E>,
}

impl<E: ExtensionField> Proof<E> {
    pub fn output_claim(&self) -> Option<Claim<E>> {
        self.claims.last().cloned()
    }
    pub fn input_claim(&self) -> Option<Claim<E>> {
        self.claims.first().cloned()
    }
}

struct Context<E: ExtensionField> {
    circuit: Circuit<E>,
    _p: PhantomData<E>,
}

pub trait LookupProtocol<E: ExtensionField> {
    // lookups has dimension (N,N) in case of two columns
    // with N lookup "op".
    // we must pad lookups MLEs to same dim than table
    // table can come from context
    // e.g table[i].num_vars() == lookups[i].num_vars()
    fn prove<T: Transcript<E>>(
        table: Vec<MLE<E>>,
        lookups: Vec<MLE<E>>,
        t: &mut T,
    ) -> anyhow::Result<Proof<E>>;

    // commitments to the lookups, one commitment per "column"
    fn verify<T: Transcript<E>>(ctx: Context<E>, proof: Proof<E>, t: &mut T) -> anyhow::Result<()>;
}

pub struct DummyLookup {}

impl<E: ExtensionField> LookupProtocol<E> for DummyLookup {
    fn prove<T: Transcript<E>>(
        table: Vec<MLE<E>>,
        lookups: Vec<MLE<E>>,
        t: &mut T,
    ) -> anyhow::Result<Proof<E>> {
        assert_eq!(table.len(), lookups.len());
        assert!(
            table
                .iter()
                .zip(lookups.iter())
                .all(|(t, l)| t.num_vars() == l.num_vars())
        );

        let claims = lookups
            .iter()
            .map(|l| {
                // TODO: to replace via proper lookup protocol output
                let point = random_field_vector(l.num_vars());
                let eval = l.evaluate(&point);
                Claim { point, eval }
            })
            .collect::<Vec<Claim<E>>>();

        let gkr_proof = IOPProof {
            sumcheck_proofs: vec![],
        };
        let output_claim = E::ONE;
        Ok(Proof {
            gkr_proof,
            output_claim,
            claims,
            _p: PhantomData,
        })
    }
    fn verify<T: Transcript<E>>(ctx: Context<E>, proof: Proof<E>, t: &mut T) -> anyhow::Result<()> {
        todo!()
    }
}
pub struct GoldilocksLogUp;

impl LookupProtocol<GoldilocksExt2> for GoldilocksLogUp {
    fn prove<T: Transcript<GoldilocksExt2>>(
        table: Vec<MLE<GoldilocksExt2>>,
        lookups: Vec<MLE<GoldilocksExt2>>,
        t: &mut T,
    ) -> anyhow::Result<Proof<GoldilocksExt2>> {
        // Check that table and lookups aren't empty
        if table.is_empty() || lookups.is_empty() {
            return Err(anyhow!("Can't prove a lookup with no table or lookups"));
        }
        // Check we have the same number of columns
        let no_columns = table.len();
        if no_columns != lookups.len() {
            return Err(anyhow!(
                "Table and lookups do not have the same number of columns"
            ));
        }
        // Check all the the columns have the same number of variables
        let num_vars = table[0].num_vars();
        for poly in table.iter().skip(1).chain(lookups.iter()) {
            if poly.num_vars() != num_vars {
                return Err(anyhow!(
                    "All columns in the table and the lookup should have the same number of variables"
                ));
            }
        }

        let circuit = logup_circuit::<GoldilocksExt2>(num_vars, no_columns);

        let (gkr_proof, output_claim) = prove_logup(&table, &lookups, &circuit, t)
            .ok_or(anyhow!("Couldn't generate LogUp GKR proof"))?;

        let point = &gkr_proof
            .sumcheck_proofs
            .last()
            .unwrap()
            .sumcheck_proof
            .point;

        let claims = lookups
            .iter()
            .map(|mle| Claim {
                point: point.clone(),
                eval: mle.evaluate(&point[..mle.num_vars()]),
            })
            .collect::<Vec<Claim<GoldilocksExt2>>>();

        Ok(Proof {
            gkr_proof,
            output_claim,
            claims,
            _p: PhantomData,
        })
    }

    fn verify<T: Transcript<GoldilocksExt2>>(
        ctx: Context<GoldilocksExt2>,
        proof: Proof<GoldilocksExt2>,
        t: &mut T,
    ) -> anyhow::Result<()> {
        let Proof {
            gkr_proof,
            output_claim,
            ..
        } = proof;

        // For now just check that we don't error.
        // TODO: add step verifying output GKR claim
        let _ = verify_logup(output_claim, gkr_proof, &ctx.circuit, t)
            .map_err(|e| anyhow!("Error verifying GKR: {:?}", e))?;
        Ok(())
    }
}
