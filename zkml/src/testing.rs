use ark_std::rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};
use ff_ext::ExtensionField;
use itertools::Itertools;

pub fn random_vector<T>(n: usize) -> Vec<T> where Standard: Distribution<T> {
    let mut rng = thread_rng();
    (0..n).map(|_| rng.gen::<T>()).collect_vec()
}

pub fn random_field_vector<E: ExtensionField>(n: usize) -> Vec<E> {
    let mut rng = thread_rng();
    (0..n).map(|_| E::random(&mut rng)).collect_vec()
}

pub fn random_bool_vector<E: ExtensionField>(n: usize) -> Vec<E> {
    let mut rng = thread_rng();
    (0..n)
        .map(|_| E::from(rng.gen_bool(0.5) as u64))
        .collect_vec()
}
