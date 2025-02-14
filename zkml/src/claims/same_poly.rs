//! This module contains logic to prove the opening of several claims related to the _same_ polynomial.
//! e.g. a set of (r_i,y_i) such that f(r_i) = y_i for all i's. That polynomial is committed during 
//! proving time (not at setup time).
//! a_i = randomness() for i:0 -> |r_i|
//! for r_i, compute Beta_{r_i} = [beta_{r_i}(0),(1),...(2^|r_i|)]
//! then Beta_j = SUM_j a_i * Beta_{r_i}

use anyhow::{ensure, Context as CC, Ok};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::virtual_poly::{VPAuxInfo, VirtualPolynomial};
use serde::{de::DeserializeOwned, Serialize};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use crate::{claims::identity_eval, vector_to_mle, VectorTranscript};
use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
use transcript::Transcript;

use super::{aggregated_rlc, compute_beta_eval_poly, compute_betas_eval, Pcs};

pub struct Context<E: ExtensionField> 
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    vp_info: VPAuxInfo<E>,
    pp: <Pcs<E> as PolynomialCommitmentScheme<E>>::ProverParam,
    vp: <Pcs<E> as PolynomialCommitmentScheme<E>>::VerifierParam,
}

impl<E: ExtensionField> Context<E> 
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// number of variables of the poly in question
    pub fn new(num_vars: usize,pcs_params: <Pcs<E> as PolynomialCommitmentScheme<E>>::Param) -> Self {
        let (pp, vp) = Pcs::trim(pcs_params, 1 << num_vars).expect("setup too small");
        Self {
            vp_info: VPAuxInfo::from_mle_list_dimensions(&[vec![num_vars, num_vars]]),
            pp,
            vp,
        }
    }
}
struct Proof<E: ExtensionField> where 
E::BaseField: Serialize + DeserializeOwned,
E: Serialize + DeserializeOwned
{
    sumcheck: IOPProof<E>,
    // [0] about the betas, [1] about the poly
    evals: Vec<E>,
    poly_comm: <Pcs<E> as PolynomialCommitmentScheme<E>>::Commitment,
    poly_open: <Pcs<E> as PolynomialCommitmentScheme<E>>::Proof,

}

struct Prover<E: ExtensionField> {
    claims: Vec<Claim<E>>,
    poly: DenseMultilinearExtension<E>,
}

impl<E> Prover<E> where 
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned {
    /// The polynomial over which the claims are to be accumulated and proven
    /// Note the prover also _commits_ to this polynomial.
    pub fn new(poly: DenseMultilinearExtension<E>) -> Self {
        Self {
            claims: Default::default(),
            poly,
        }
    }
    pub fn add_claim(&mut self,input: Vec<E>, output: E) -> anyhow::Result<()> {
        ensure!(input.len() == self.poly.num_vars(),
            format!("Invalid claim length: input.len() = {} vs poly.num_vars = {} ",
            input.len(),self.poly.num_vars()));
        self.claims.push(Claim {
            input,
            output,
        });
        Ok(())
    }
    pub fn prove<T: Transcript<E>>(self,ctx: &Context<E>, t: &mut T) -> anyhow::Result<Proof<E>> {
        let challenges = t.read_challenges(self.claims.len());
        //#[cfg(test)]
        //{
        //    let inputs = self.claims.into_iter().map(|c| (c.input,c.output)).unzip();
        //    let y_res = aggregated_rlc(&outputs, &challenges);
        //}

        let mut final_beta = vec![E::ZERO; 1 << self.poly.num_vars()];
        // TODO: see if this methods could run faster with multiple threads instead using matmul in parallel
        for (a_i,c_i) in challenges.into_iter().zip(self.claims) {
            // c_i.input = r_i
            let beta_i = compute_betas_eval(&c_i.input);
            // beta_i(j)
            for (j,beta_i_j) in beta_i.into_iter().enumerate() {
                final_beta[j] += a_i * beta_i_j;
            }
        }
        // then run the sumcheck on it
        let mut vp = VirtualPolynomial::new(self.poly.num_vars());
        vp.add_mle_list(vec![vector_to_mle(final_beta).into(),self.poly.clone().into()], E::ONE);
         #[allow(deprecated)]
        let (sumcheck_proof, state) = IOPProverState::<E>::prove_parallel(vp, t);

        // prove the opening for the polynomial at the random point from the sumcheck
        let poly_evaluation = state.get_mle_final_evaluations()[1];
        let poly_point = sumcheck_proof.point.clone();
        let comm = Pcs::commit(&ctx.pp, &self.poly).context("unable to commit")?;
        let vcomm = Pcs::get_pure_commitment(&comm);
        let pcs_proof = Pcs::open(&ctx.pp, &self.poly, &comm, &poly_point, &poly_evaluation, t)?;
        Ok(Proof{
            sumcheck: sumcheck_proof,
            evals: state.get_mle_final_evaluations(),
            poly_comm: vcomm,
            poly_open: pcs_proof,
        })
    }
}

struct Verifier<'a, E: ExtensionField> 
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    claims: Vec<Claim<E>>,
    ctx: &'a Context<E>,
}

impl<'a, E: ExtensionField> Verifier<'a, E> 
where 
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned
{
    pub fn new(ctx: &'a Context<E>) -> Self {
        Self {
            claims: Default::default(),
            ctx,
        }
    }
    
    pub fn add_claim(&mut self, input: Vec<E>, output: E) -> anyhow::Result<()> {
        ensure!(input.len() == self.ctx.vp_info.max_num_variables,"invalid input len wrt to poly in ctx");
        self.claims.push(Claim {
            input,
            output,
        });
        Ok(())
    }

    pub fn verify<T: Transcript<E>>(self, proof: Proof<E>, t: &mut T) -> anyhow::Result<()> {
        let fs_challenges = t.read_challenges(self.claims.len()) ;
        let (rs,ys) :(Vec<_>,Vec<_>)= self.claims.into_iter().map(|c| (c.input,c.output)).unzip();
        let y_res = aggregated_rlc(&ys, &fs_challenges);
        // check sumcheck proof
        let subclaim = IOPVerifierState::<E>::verify(y_res, &proof.sumcheck, &self.ctx.vp_info, t);
        // check sumcheck output: first check for the betas we can compute
        //for(int i = 0; i < a.size(); i++){y += a[i]*identity_eval(claims[i].first,P.randomness[0]);}
        let computed_y = fs_challenges.into_iter().zip(rs).fold(E::ZERO,|acc,(a_i,r_i)| {
            acc + a_i * identity_eval(&r_i, &proof.sumcheck.point)
        });
        let given_y = proof.evals[0];
        ensure!(computed_y == given_y,"beta evaluation do not match");
        // then check opening proof for the part about the poly
        let point = proof.sumcheck.point.clone();
        let eval = proof.evals[1];
        Pcs::verify(&self.ctx.vp, &proof.poly_comm, &point, &eval, &proof.poly_open, t)
            .context("invalid pcs opening")?;

        // then check that both betas and poly evaluation lead to the outcome of the sumcheck, e.g. the sum
        let expected = proof.evals[0] * proof.evals[1];
        let computed = subclaim.expected_evaluation;
        ensure!(expected == computed,"final evals of sumcheck is not valid");
        Ok(())
    }
}

/// Claim type to accumulate in this protocol
struct Claim<E> {
    input: Vec<E>,
    output: E,
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;
    use mpcs::PolynomialCommitmentScheme;
    use multilinear_extensions::mle::MultilinearExtension;

    use crate::{claims::Pcs, model::test::random_vector, prover::default_transcript, vector_to_mle};
    use itertools::Itertools;

    use super::{Context, Prover, Verifier};

    type F = GoldilocksExt2;

    #[test]
    fn test_pcs() {
        let num_vars = 10;
        let len = 1 << num_vars;
        let _param = Pcs::<F>::setup(len).expect("unable to setup");
    }

    #[test]
    fn test_same_poly_proof() -> anyhow::Result<()> {
        // number of vars
        let num_vars = 10 as usize;
        let poly_len = 1 << num_vars;
        let poly = random_vector::<F>(poly_len);
        let poly_mle = vector_to_mle(poly.clone());
        // number of clains
        let m = 14;
        let claims = (0..m).map(|_| {
            let r_i = random_vector(num_vars);
            let y_i = poly_mle.evaluate(&r_i);
            (r_i,y_i)
        }).collect_vec();
        // COMMON PART
        assert_eq!(poly.len(), 1 << num_vars);
        let param = Pcs::setup(poly_len)?;
        let ctx = Context::new(num_vars,param);
        // PROVER PART
        let mut t = default_transcript();
        let mut prover = Prover::new(poly_mle);
        for (r_i,y_i) in claims.clone().into_iter() {
            prover.add_claim(r_i, y_i)?;
        }
        let proof = prover.prove(&ctx,&mut t)?;
        // VERIFIER PART
        let mut t = default_transcript();
        let mut verifier = Verifier::new(&ctx);
        for (r_i, y_i) in claims.into_iter() {
            verifier.add_claim(r_i, y_i)?;
        }
        verifier.verify(proof, &mut t)?;
        Ok(())
    }
}