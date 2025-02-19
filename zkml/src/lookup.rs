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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierClaims<E: ExtensionField> {
    input_claims: Vec<Claim<E>>,
    output_claims: Vec<Claim<E>>,
    multiplicity_claim: Claim<E>,
}

impl<E: ExtensionField> VerifierClaims<E> {
    pub fn input_claims(&self) -> &[Claim<E>] {
        &self.input_claims
    }

    pub fn output_claims(&self) -> &[Claim<E>] {
        &self.output_claims
    }

    pub fn multiplicity_poly_claim(&self) -> &Claim<E> {
        &self.multiplicity_claim
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
    fn verify<T: Transcript<E>>(
        ctx: Context<E>,
        proof: Proof<E>,
        t: &mut T,
    ) -> anyhow::Result<VerifierClaims<E>>;
}


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
                    point: final_point[..mle.num_vars()].to_vec(),
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
                    point: final_point[..mle.num_vars()].to_vec(),
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

    fn verify<T: Transcript<E>>(
        ctx: Context<E>,
        proof: Proof<E>,
        t: &mut T,
    ) -> anyhow::Result<VerifierClaims<E>> {
        let Proof {
            gkr_proof,
            output_claim: denominator_product,
            ..
        } = proof;

        let number_table_columns = ctx.no_input_columns + ctx.no_output_columns;
        // Verify the GKR proof, this involves verifying that `denominator_product` is non-zero and outputs claims about
        // the polynomials used in the GKR circuit. these claims will be ordered as table_polys, lookup_polys, multiplicity_poly
        let gkr_claims = verify_logup(denominator_product, gkr_proof, ctx.circuit(), t)
            .map_err(|e| anyhow!("Error verifying GKR: {:?}", e))?;

        let input_claims = gkr_claims.point_and_evals
            [number_table_columns..number_table_columns + ctx.no_input_columns]
            .iter()
            .map(Claim::from)
            .collect::<Vec<Claim<E>>>();
        let output_claims = gkr_claims.point_and_evals
            [number_table_columns + ctx.no_input_columns..2 * number_table_columns]
            .iter()
            .map(Claim::from)
            .collect::<Vec<Claim<E>>>();
        let multiplicity_claim =
            Claim::<E>::from(&gkr_claims.point_and_evals[2 * number_table_columns]);

        Ok(VerifierClaims {
            input_claims,
            output_claims,
            multiplicity_claim,
        })
    }
}
