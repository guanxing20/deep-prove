//! This module contains logic to prove the opening of several claims related to the _same_ polynomial.
//! e.g. a set of (r_i,y_i) such that f(r_i) = y_i for all i's. That polynomial is committed during 
//! proving time (not at setup time).
//! a_i = randomness() for i:0 -> |r_i|
//! for r_i, compute Beta_{r_i} = [beta_{r_i}(0),(1),...(2^|r_i|)]
//! then Beta_j = SUM_j a_i * Beta_{r_i}

use anyhow::{ensure, Ok};
use ff_ext::ExtensionField;
use multilinear_extensions::virtual_poly::{VPAuxInfo, VirtualPolynomial};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use tract_onnx::prelude::tract_linalg::frame::reduce::sum;
use crate::{vector_to_mle, VectorTranscript};
use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
use transcript::Transcript;

use super::{aggregated_rlc, compute_betas_eval};

pub struct Context<E> {
    vp_info: VPAuxInfo<E> 
}

impl<E> Context<E> {
    /// number of variables of the poly in question
    pub fn new(num_vars: usize) -> Self {
        Self {
            vp_info: VPAuxInfo::from_mle_list_dimensions(&[vec![num_vars, num_vars]]),
        }
    }
}
struct Proof<E: ExtensionField> {
    sumcheck: IOPProof<E>,
    evals: Vec<E>,
}

struct Prover<E: ExtensionField> {
    claims: Vec<Claim<E>>,
    poly: DenseMultilinearExtension<E>,
}

impl<E> Prover<E> where E: ExtensionField {
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
    pub fn prove<T: Transcript<E>>(self,t: &mut T) -> anyhow::Result<Proof<E>> {
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
        vp.add_mle_list(vec![vector_to_mle(final_beta).into(),self.poly.into()], E::ONE);
         #[allow(deprecated)]
        let (sumcheck_proof, state) = IOPProverState::<E>::prove_parallel(vp, t);
        Ok(Proof{
            sumcheck: sumcheck_proof,
            evals: state.get_mle_final_evaluations(),
        })
    }
}

struct Verifier<'a, E> {
    claims: Vec<Claim<E>>,
    ctx: &'a Context<E>,
}

impl<'a, E: ExtensionField> Verifier<'a, E> {
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
        let challenges = t.read_challenges(self.claims.len()) ;
        let (rs,ys) :(Vec<_>,Vec<_>)= self.claims.into_iter().map(|c| (c.input,c.output)).unzip();
        let y_res = aggregated_rlc(&ys, &challenges);
        // check sumcheck 
        let subclaim = IOPVerifierState::<E>::verify(y_res, &proof.sumcheck, &self.ctx.vp_info, t);
        // then check opening
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
    use multilinear_extensions::mle::MultilinearExtension;

    use crate::{model::test::random_vector, prover::default_transcript, vector_to_mle};
    use itertools::Itertools;

    use super::{Context, Prover, Verifier};

    type F = GoldilocksExt2;

    #[test]
    fn test_same_poly_proof() -> anyhow::Result<()> {
        // number of vars
        let n = (10 as usize);
        let poly = random_vector::<F>(1 << n);
        let poly_mle = vector_to_mle(poly.clone());
        // number of clains
        let m = 14;
        let claims = (0..m).map(|_| {
            let r_i = random_vector(n);
            let y_i = poly_mle.evaluate(&r_i);
            (r_i,y_i)
        }).collect_vec();
        // COMMON PART
        let ctx = Context::new(n);
        // PROVER PART
        let mut t = default_transcript();
        let mut prover = Prover::new(poly_mle);
        for (r_i,y_i) in claims.clone().into_iter() {
            prover.add_claim(r_i, y_i)?;
        }
        let proof = prover.prove(&mut t)?;
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