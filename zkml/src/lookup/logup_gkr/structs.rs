//! Module containing utility structs for working with LogUp GKR circuits.

use std::{
    borrow::Borrow,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign},
};

use ff_ext::ExtensionField;
use multilinear_extensions::mle::DenseMultilinearExtension;
use serde::{Deserialize, Serialize};
use sumcheck::structs::IOPProof;
use transcript::Transcript;

use super::circuit::LogUpCircuit;
use crate::Claim;
use rayon::prelude::*;

#[derive(Clone, Debug, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
/// Struct used to perform arithmetic on fractions
pub struct Fraction<F> {
    pub numerator: F,
    pub denominator: F,
}

impl<F> Fraction<F> {
    /// Create a new instance of a [`Fraction`].
    pub fn new(numerator: F, denominator: F) -> Fraction<F> {
        Fraction::<F> {
            numerator,
            denominator,
        }
    }

    /// Turns this fraction into a tuple, the first element is the numerator, the second is the denominator
    pub fn as_tuple(&self) -> (F, F)
    where
        F: Clone,
    {
        (self.numerator.clone(), self.denominator.clone())
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> AddAssign<T> for Fraction<F> {
    fn add_assign(&mut self, rhs: T) {
        let rhs: &Fraction<F> = rhs.borrow();
        let numerator = self.numerator * rhs.denominator + self.denominator * rhs.numerator;
        let denominator = self.denominator * rhs.denominator;
        *self = Fraction {
            numerator,
            denominator,
        };
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> Add<T> for &Fraction<F> {
    type Output = Fraction<F>;

    fn add(self, rhs: T) -> Self::Output {
        let mut output = *self;
        output += rhs;
        output
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> Add<T> for Fraction<F> {
    type Output = Fraction<F>;

    fn add(self, rhs: T) -> Self::Output {
        let mut output = self;
        output += rhs;
        output
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> MulAssign<T> for Fraction<F> {
    fn mul_assign(&mut self, rhs: T) {
        let rhs: &Fraction<F> = rhs.borrow();
        self.numerator *= rhs.numerator;
        self.denominator *= rhs.denominator;
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> Mul<T> for &Fraction<F> {
    type Output = Fraction<F>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut output = *self;
        output *= rhs;
        output
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> Mul<T> for Fraction<F> {
    type Output = Fraction<F>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut output = self;
        output *= rhs;
        output
    }
}

impl<F: ExtensionField> Sum for Fraction<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Fraction<F> {
        iter.fold(Fraction::<F>::ZERO, |acc, term| acc + term)
    }
}

impl<F: ExtensionField> Product for Fraction<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Fraction::<F>::ONE, |acc, term| acc * term)
    }
}

impl<F: ExtensionField> Fraction<F> {
    const ZERO: Fraction<F> = Fraction {
        numerator: F::ZERO,
        denominator: F::ONE,
    };

    const ONE: Fraction<F> = Fraction {
        numerator: F::ONE,
        denominator: F::ONE,
    };

    pub fn is_zero(&self) -> bool {
        (self.numerator == F::ZERO) && (self.denominator != F::ZERO)
    }
}

#[derive(Clone, Debug)]
/// Struct that represents a lookup variant input to the LogUp GKR protocol
pub struct LookupInput<E: ExtensionField> {
    column_evals: Vec<Vec<E::BaseField>>,
    constant_challenge: E,
    column_separation_challenge: E,
    num_columns_per_instance: usize,
}

impl<E: ExtensionField> LookupInput<E> {
    /// Creates a new instance of [`LookupInput`]
    pub fn new(
        column_evals: Vec<Vec<E::BaseField>>,
        constant_challenge: E,
        column_separation_challenge: E,
        num_columns_per_instance: usize,
    ) -> LookupInput<E> {
        LookupInput {
            column_evals,
            constant_challenge,
            column_separation_challenge,
            num_columns_per_instance,
        }
    }

    /// Produces the [`LogUpCircuit`]s from this instance
    pub fn make_circuits(&self) -> Vec<LogUpCircuit<E>> {
        self.column_evals
            .par_chunks(self.num_columns_per_instance)
            .map(|column_evals| {
                LogUpCircuit::<E>::new_lookup_circuit(
                    column_evals,
                    self.constant_challenge,
                    self.column_separation_challenge,
                )
            })
            .collect()
    }

    /// Get the base mles
    pub fn base_mles(&self) -> Vec<DenseMultilinearExtension<E>> {
        self.column_evals
            .iter()
            .map(|evaluations| {
                let num_vars = evaluations.len().ilog2() as usize;
                DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, evaluations)
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
/// Struct that represents a table variant input to the LogUp GKR protocol
pub struct TableInput<E: ExtensionField> {
    column_evals: Vec<Vec<E::BaseField>>,
    multiplicities: Vec<E::BaseField>,
    constant_challenge: E,
    column_separation_challenge: E,
}

impl<E: ExtensionField> TableInput<E> {
    /// Create a new [`TableInput`]
    pub fn new(
        column_evals: Vec<Vec<E::BaseField>>,
        multiplicities: Vec<E::BaseField>,
        constant_challenge: E,
        column_separation_challenge: E,
    ) -> TableInput<E> {
        TableInput {
            column_evals,
            multiplicities,
            constant_challenge,
            column_separation_challenge,
        }
    }

    /// Produces the [`LogUpCircuit`]s from this instance
    pub fn make_circuit(&self) -> LogUpCircuit<E> {
        LogUpCircuit::<E>::new_table_circuit(
            &self.column_evals,
            &self.multiplicities,
            self.constant_challenge,
            self.column_separation_challenge,
        )
    }

    /// Returns the underlying MLEs in the order `multiplicities`, `columns`
    pub fn base_mles(&self) -> Vec<DenseMultilinearExtension<E>> {
        std::iter::once(&self.multiplicities)
            .chain(self.column_evals.iter())
            .map(|evaluations| {
                let num_vars = evaluations.len().ilog2() as usize;
                DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, evaluations)
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct LogUpProof<E: ExtensionField> {
    pub sumcheck_proofs: Vec<IOPProof<E>>,
    pub round_evaluations: Vec<Vec<E>>,
    pub output_claims: Vec<Claim<E>>,
    pub circuit_outputs: Vec<Vec<E>>,
}

impl<E: ExtensionField> LogUpProof<E> {
    pub fn append_to_transcript<T: Transcript<E>>(&self, transcript: &mut T) {
        self.circuit_outputs
            .iter()
            .for_each(|evals| transcript.append_field_element_exts(evals));
    }

    pub fn fractional_outputs(&self) -> (Vec<E>, Vec<E>) {
        self.circuit_outputs
            .iter()
            .map(|evals| {
                (
                    evals[0] * evals[2] + evals[1] * evals[3],
                    evals[2] * evals[3],
                )
            })
            .unzip()
    }

    pub fn proofs_and_evals(&self) -> impl Iterator<Item = (&IOPProof<E>, &Vec<E>)> {
        self.sumcheck_proofs
            .iter()
            .zip(self.round_evaluations.iter())
    }

    pub fn circuit_outputs(&self) -> &[Vec<E>] {
        &self.circuit_outputs
    }

    pub fn output_claims(&self) -> &[Claim<E>] {
        &self.output_claims
    }
}
