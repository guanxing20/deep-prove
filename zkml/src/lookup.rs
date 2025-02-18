use std::marker::PhantomData;

use anyhow::anyhow;
use ark_std::rand::thread_rng;
use ff_ext::ExtensionField;
use gkr::structs::{Circuit, IOPProof};

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
    input_column_claims: Vec<Claim<E>>,
    output_column_claims: Vec<Claim<E>>,
    m_claim: Claim<E>,
    gkr_proof: IOPProof<E>,
    output_claim: E,
    // multi_C: Commitment<E>,
}

impl<E: ExtensionField> Proof<E> {
    /// Retireve the claims about the input columns
    pub fn input_column_claims(&self) -> &[Claim<E>] {
        &self.input_column_claims
    }

    /// Retrieve the output column claims
    pub fn output_column_claims(&self) -> &[Claim<E>] {
        &self.output_column_claims
    }

    /// Retrieve the multiplicity poly claim
    pub fn multiplicity_claim(&self) -> &Claim<E> {
        &self.m_claim
    }
}

pub struct Context<E: ExtensionField> {
    circuit: Circuit<E>,
    no_input_columns: usize,
    no_output_columns: usize,
}

impl<E: ExtensionField> Context<E> {
    /// Getter for the [`Circuit`]
    pub fn circuit(&self) -> &Circuit<E> {
        &self.circuit
    }

    /// Builds a new context
    pub fn new(
        table_num_vars: usize,
        lookup_num_vars: usize,
        no_input_columns: usize,
        no_output_columns: usize,
    ) -> Self {
        let circuit = logup_circuit::<E>(
            table_num_vars,
            lookup_num_vars,
            no_input_columns + no_output_columns,
        );

        Self {
            circuit,
            no_input_columns,
            no_output_columns,
        }
    }
}

pub trait LookupProtocol<E: ExtensionField> {
    // lookups has dimension (N,N) in case of two columns
    // with N lookup "op".
    // we must pad lookups MLEs to same dim than table
    // table can come from context
    // e.g table[i].num_vars() == lookups[i].num_vars()
    fn prove<T: Transcript<E>>(
        ctx: &Context<E>,
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
        ctx: &Context<E>,
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

        let input_column_claims = lookups
            .iter()
            .take(ctx.no_input_columns)
            .map(|l| {
                // TODO: to replace via proper lookup protocol output
                let point = random_field_vector(l.num_vars());
                let eval = l.evaluate(&point);
                Claim { point, eval }
            })
            .collect::<Vec<Claim<E>>>();

        let output_column_claims = lookups
            .iter()
            .skip(ctx.no_input_columns)
            .take(ctx.no_output_columns)
            .map(|l| {
                // TODO: to replace via proper lookup protocol output
                let point = random_field_vector(l.num_vars());
                let eval = l.evaluate(&point);
                Claim { point, eval }
            })
            .collect::<Vec<Claim<E>>>();

        let m_claim = Claim {
            point: random_field_vector(table[0].num_vars()),
            eval: E::random(thread_rng()),
        };

        let gkr_proof = IOPProof {
            sumcheck_proofs: vec![],
        };
        let output_claim = E::ONE;
        Ok(Proof {
            gkr_proof,
            output_claim,
            input_column_claims,
            output_column_claims,
            m_claim,
        })
    }
    fn verify<T: Transcript<E>>(ctx: Context<E>, proof: Proof<E>, t: &mut T) -> anyhow::Result<()> {
        todo!()
    }
}
pub struct GoldilocksLogUp;
pub struct LogUp<E: ExtensionField>(PhantomData<E>);

impl<E: ExtensionField> LookupProtocol<E> for LogUp<E> {
    fn prove<T: Transcript<E>>(
        ctx: &Context<E>,
        table: Vec<MLE<E>>,
        lookups: Vec<MLE<E>>,
        t: &mut T,
    ) -> anyhow::Result<Proof<E>> {
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

        // Make sure that the table columns have the same number of variables
        table.windows(2).try_for_each(|window| {
            if window[0].num_vars() == window[1].num_vars() {
                Ok(())
            } else {
                Err(anyhow!(
                    "Not all provided table MLEs have the same number of variables"
                ))
            }
        })?;
        // Make sure that the lookup columns have the same number of variables
        lookups.windows(2).try_for_each(|window| {
            if window[0].num_vars() == window[1].num_vars() {
                Ok(())
            } else {
                Err(anyhow!(
                    "Not all provided lookup MLEs have the same number of variables"
                ))
            }
        })?;

        // Here the `output_claim` is the product of all the denominators in the fractional sumcheck, the verifier checks that this value is non-zero.
        let (gkr_proof, output_claim, multiplicity_poly) =
            prove_logup(&table, &lookups, ctx.circuit(), t)
                .ok_or(anyhow!("Couldn't generate LogUp GKR proof"))?;

        let final_point = &gkr_proof
            .sumcheck_proofs
            .last()
            .unwrap()
            .sumcheck_proof
            .point;

        let input_column_claims = lookups
            .iter()
            .take(ctx.no_input_columns)
            .map(|mle| {
                let eval = mle.evaluate(&final_point[..mle.num_vars()]);
                Claim {
                    point: final_point.clone(),
                    eval,
                }
            })
            .collect::<Vec<Claim<E>>>();

        let output_column_claims = lookups
            .iter()
            .skip(ctx.no_input_columns)
            .take(ctx.no_output_columns)
            .map(|mle| {
                let eval = mle.evaluate(&final_point[..mle.num_vars()]);
                Claim {
                    point: final_point.clone(),
                    eval,
                }
            })
            .collect::<Vec<Claim<E>>>();

        let m_claim = Claim {
            point: final_point[..multiplicity_poly.num_vars()].to_vec(),
            eval: multiplicity_poly.evaluate(&final_point[..multiplicity_poly.num_vars()]),
        };

        Ok(Proof {
            gkr_proof,
            output_claim,
            input_column_claims,
            output_column_claims,
            m_claim,
        })
    }

    fn verify<T: Transcript<E>>(ctx: Context<E>, proof: Proof<E>, t: &mut T) -> anyhow::Result<()> {
        let Proof {
            gkr_proof,
            output_claim,
            ..
        } = proof;

        // For now just check that we don't error.
        // TODO: add step verifying output GKR claim
        let _ = verify_logup(output_claim, gkr_proof, ctx.circuit(), t)
            .map_err(|e| anyhow!("Error verifying GKR: {:?}", e))?;
        Ok(())
    }
}
