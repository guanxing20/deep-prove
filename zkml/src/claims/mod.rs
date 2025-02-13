use ff_ext::ExtensionField;

pub mod precommit;
pub mod same_poly;

/// Compute the vector (beta(r,1), ... ,beta(r,2^{|r|}))
/// This function uses the dynamic programing technique of Libra
pub fn compute_betas_eval<E: ExtensionField>(r: &[E]) -> Vec<E> {
    let n = r.len();
    let size = 1 << n;
    let mut betas = vec![E::ZERO; size];
    betas[0] = E::ONE;

    for i in 0..n {
        let current_size = 1 << i;
        let temp = betas[..current_size].to_vec();
        let r_elem = r[r.len() - 1 - i];
        for j in 0..current_size {
            let idx = j << 1;
            let t = r_elem * temp[j];
            betas[idx] = temp[j] - t;
            betas[idx + 1] = t;
        }
    }
    betas
}

/// Random linear combination of claims and random elements derived from transcript
pub fn aggregated_rlc<E: ExtensionField>(claims: &[E], challenges: &[E]) -> E {
    assert_eq!(claims.len(), challenges.len());
    claims
        .iter()
        .zip(challenges)
        .fold(E::ZERO, |acc, (claim, r)| acc + *claim * r)
}