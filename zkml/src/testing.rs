use ark_std::rand::{thread_rng, Rng};
use ff_ext::ExtensionField;
use itertools::Itertools;

use crate::Element;

pub fn random_vector(n: usize) -> Vec<Element> {
        let mut rng = thread_rng();
        (0..n).map(|_| rng.gen::<u8>() as Element).collect_vec()
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