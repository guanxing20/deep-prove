//! Module that takes care of (re)quantizing 

use ff_ext::ExtensionField;
use goldilocks::SmallField;
use serde::{Deserialize, Serialize};

use crate::{matrix::Matrix, Element};

pub type QuantInteger = i8;
pub const BIT_LEN :usize = QuantInteger::BITS as usize;
pub const MAX: QuantInteger = QuantInteger::MAX;
pub const MIN: QuantInteger = QuantInteger::MIN;
pub const ZERO: QuantInteger = 0;

/// Trait used to quantize original floating point number to integer
pub(crate) trait Quantizer<Output> {
    fn from_f32_unsafe(e: &f32) -> Output;
}

pub(crate) trait Fieldizer<F> {
    fn to_field(&self) -> F;
}

impl Quantizer<QuantInteger> for QuantInteger {
    fn from_f32_unsafe(e: &f32) -> Self {
        let max = QuantInteger::MAX;
        // (a -b) / 2^Q
        let scale = (1.0 - (-1.0)) / max as f64;
        let zero_point = 0;

        // formula is q = round(r/S) + z
        let scaled = (*e as f64 / scale).round() as u32 + zero_point;
        scaled as QuantInteger
    }
}

impl<F: ExtensionField> Fieldizer<F> for QuantInteger {
    fn to_field(&self) -> F {
        if self.is_negative() {
            // Doing wrapped arithmetic : p-128 ... p-1 means negative number
            F::from(<F::BaseField as SmallField>::MODULUS_U64 - self.abs() as u64)
        } else {
            // for positive and zero, it's just the number
            F::from(*self as u64)
        }
    }
}
/// BIT_LEN is the target bit length we want to reduce any number to
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct QuantInfo<const BIT_LEN: usize> {
    // a - b: at the beginning a power of two to simplify requantization
    pub(crate) max_range: usize,
}


impl<const BIT_LEN:usize > QuantInfo<BIT_LEN> {
    pub fn default() -> Self {
        QuantInfo {
            max_range: (2 as usize).pow(BIT_LEN as u32),
        }
    }
    /// Computes the quantization info that a matrix x vec will produce
    /// self should be the quant info of the matrix
    /// The quantization info is depending on the number of columns in the matrix
    /// NOTE: this is assuming the vector has the same quantization factor as the matrix coeff
    /// NOTE2: It is using the simplfiication of finding the max range which is a power of two
    /// so we only need to "right shift" during requant
    pub fn compute_matvec_quant(&self, m: &Matrix<Element>) -> Self {
        // BIT_LEN * 2 because of multiplication
        // log because of additions
        let bit_len = BIT_LEN * 2  + m.ncols().ilog2() as usize;
        Self {
            max_range: (2 as usize).pow(bit_len as u32),
        }
    }
    /// Computes the right shift required to perform after multiplying two numbers
    pub fn mult_shift(self, rhs: QuantInfo<BIT_LEN>,output: QuantInfo<BIT_LEN>) -> usize {
        assert!(self.max_range.is_power_of_two());
        assert!(rhs.max_range.is_power_of_two());
        assert!(output.max_range.is_power_of_two());
        let slog = self.max_range.ilog2() as usize;
        let rlog = rhs.max_range.ilog2() as usize;
        let olog = output.max_range.ilog2() as usize;
        slog + rlog - olog - BIT_LEN
    }
}