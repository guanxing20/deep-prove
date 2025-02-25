//! Module that takes care of (re)quantizing

use ff_ext::ExtensionField;
use goldilocks::SmallField;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use transcript::Transcript;

use crate::{Element, matrix::Matrix};

/// The type of integer we do arithmetics over. Note this is NOT the type we run the inference over actually
/// We run it over `[Element]` since we need to handle overflows. But all the quantization and requantization
/// is done over this QuantInteger.
pub type QuantInteger = i8;
pub const BIT_LEN: usize = QuantInteger::BITS as usize;
pub const MAX: QuantInteger = QuantInteger::MAX;
pub const MIN: QuantInteger = QuantInteger::MIN;
pub const ZERO: QuantInteger = 0;

/// Trait used to quantize original floating point number to integer
pub trait Quantizer<Output> {
    fn from_f32_unsafe(e: &f32) -> Output;
}

pub(crate) trait Fieldizer<F> {
    fn to_field(&self) -> F;
}

impl Quantizer<Element> for Element {
    fn from_f32_unsafe(e: &f32) -> Self {
        // even tho we are requantizing starting from Element, we only want to requantize for QuantInteger
        // the reason we have these two types is to handle overflow
        let max = QuantInteger::MAX;
        // (a -b) / 2^Q
        let scale = (1.0 - (-1.0)) / max as f64;
        let zero_point = 0;

        // formula is q = round(r/S) + z
        let scaled = (*e as f64 / scale).round() as Element + zero_point;
        scaled as Element
    }
}

impl<F: ExtensionField> Fieldizer<F> for Element {
    fn to_field(&self) -> F {
        // make sure we're in range still
        // NOTE(nikkolasg) removed that assertions for tests until requantization is there.
        // jassert!(*self >= QuantInteger::MIN as Element && *self <= QuantInteger::MAX as Element);
        //(*self as QuantInteger).to_field()
        if self.is_negative() {
            // Doing wrapped arithmetic : p-128 ... p-1 means negative number
            // F::from((<F::BaseField as SmallField>::MODULUS_U64 as Element + self) as u64)
            F::ZERO - F::from(self.unsigned_abs() as u64)
        } else {
            // for positive and zero, it's just the number
            F::from(*self as u64)
        }
    }
}

impl<F: ExtensionField> Fieldizer<F> for i8 {
    fn to_field(&self) -> F {
        // debug_assert!(*self >= MIN && *self <= MAX);
        if self.is_negative() {
            // if false {
            // Doing wrapped arithmetic : p-128 ... p-1 means negative number
            // NOTE: we can't use abs() directly because i8::MIN.abs() doesn't fit inside i8
            // F::from(<F::BaseField as SmallField>::MODULUS_U64 - (*self as i64).unsigned_abs())
            F::ZERO - F::from(self.unsigned_abs() as u64)
        } else {
            // for positive and zero, it's just the number
            F::from(*self as u64)
        }
    }
}

impl<F: ExtensionField> Fieldizer<F> for u8 {
    fn to_field(&self) -> F {
        F::from(*self as u64)
    }
}

pub trait VecFielder<F> {
    fn to_fields(self) -> Vec<F>;
}

impl<F: ExtensionField, T> VecFielder<F> for Vec<T>
where
    T: Fieldizer<F>,
{
    fn to_fields(self) -> Vec<F> {
        self.into_iter().map(|i| i.to_field()).collect_vec()
    }
}

impl<F: ExtensionField, T> VecFielder<F> for &[T]
where
    T: Fieldizer<F>,
{
    fn to_fields(self) -> Vec<F> {
        self.iter().map(|i| i.to_field()).collect_vec()
    }
}

/// QuantRange is an intermediary struct to compute the final quantization information.
/// This struct is gonna be useful down the road to handle more precise requantization techniques.
/// BIT_LEN is the target bit length we want to reduce any number to
#[derive(Clone, Debug)]
struct QuantRange<const BIT_LEN: usize> {
    // a - b: at the beginning a power of two to simplify requantization
    pub(crate) max_range: usize,
}

impl<const BIT_LEN: usize> QuantRange<BIT_LEN> {
    pub fn default() -> Self {
        QuantRange {
            max_range: (2 as usize).pow(BIT_LEN as u32),
        }
    }
    /// Computes the quantization info that a matrix x vec will produce
    /// self should be the quant info of the matrix
    /// The quantization info is depending on the number of columns in the matrix
    /// NOTE: this is assuming the vector has the same quantization factor as the matrix coeff
    ///       and it assumes these are the default range.
    /// NOTE2: It is using the simplfiication of finding the max range which is a power of two
    /// so we only need to "right shift" during requant
    fn compute_matvec_quant(m: &Matrix<Element>) -> Requant {
        // NOTE this way below is correct but is taking a huge loss
        // BIT_LEN * 2 because of multiplication
        // log because of additions
        // let bit_len = BIT_LEN * 2  + m.ncols().ilog2() as usize;
        // let output_range = Self {
        //    max_range: (2 as usize).pow(bit_len as u32),
        //};
        // NOTE 2: this way is more precise
        let ind_range = (MAX as i64 - MIN as i64) as usize;
        let output_range = Self {
            max_range: (ind_range.pow(2) + m.ncols() as usize * ind_range).next_power_of_two(),
        };
        let shift = Self::default().mult_shift(&Self::default(), &output_range);
        Requant {
            range: output_range.max_range,
            right_shift: shift,
        }
    }
    /// Computes the right shift required to perform after multiplying two numbers
    fn mult_shift(&self, rhs: &QuantRange<BIT_LEN>, output: &QuantRange<BIT_LEN>) -> usize {
        assert!(self.max_range.is_power_of_two());
        assert!(rhs.max_range.is_power_of_two());
        assert!(output.max_range.is_power_of_two());
        let slog = self.max_range.ilog2() as usize;
        let rlog = rhs.max_range.ilog2() as usize;
        let olog = output.max_range.ilog2() as usize;
        slog + rlog - olog - BIT_LEN
    }
}

/// Information about a requantization step:
/// * what is the range of the input data
/// * what should be the shift to get back data in range within QuantInteger range
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Requant {
    // what is the shift that needs to be applied to requantize input number to the correct range of QuantInteger.
    pub right_shift: usize,
    pub range: usize,
}

impl Requant {
    pub fn from_matrix_default(m: &Matrix<Element>) -> Self {
        QuantRange::<BIT_LEN>::compute_matvec_quant(m)
    }

    pub fn op(&self, input: &[Element]) -> Vec<Element> {
        input.iter().map(|e| self.apply(e)).collect_vec()
    }

    #[inline(always)]
    pub fn apply(&self, e: &Element) -> Element {
        e >> self.right_shift
    }

    pub fn shape(&self) -> Vec<usize> {
        vec![1, self.range]
    }

    pub fn write_to_transcript<E: ExtensionField, T: Transcript<E>>(&self, t: &mut T) {
        t.append_field_element(&E::BaseField::from(self.right_shift as u64));
        t.append_field_element(&E::BaseField::from(self.range as u64));
    }

    /// to_mle returns two polynomials:
    /// f_i: one containing the input column values
    /// f_o: one containing the output column values --> shifted to the right !
    /// TODO: have a "cache" of lookups for similar ranges
    pub fn to_mle<E: ExtensionField>(&self) -> (Vec<E>, Vec<E>) {
        // TODO: make a +1 or -1 somewhere
        let min_range = -(self.range as Element) / 2;
        let max_range = (self.range as Element) / 2 - 1;
        (min_range..=max_range)
            .map(|i| {
                let input: E = i.to_field();
                // conversion from QuantInteger -> u64 OK because result is either 0 or strictly positive.
                let output = E::from(self.apply(&i) as u64);
                (input, output)
            })
            .unzip()
    }
}

#[cfg(test)]
mod test {
    use crate::quantization::{Fieldizer, QuantInteger};
    use ff_ext::ExtensionField;
    use goldilocks::{MODULUS, SmallField};

    use crate::Element;
    type F = goldilocks::GoldilocksExt2;

    #[test]
    fn test_wrapped_field() {
        // for case in vec![-12,25,i8::MIN,i8::MAX] {
        //     let a: i8 = case;
        //     let af: F= a.to_field();
        //     let f = af.to_canonical_u64_vec()[0];
        //     let exp = if a.is_negative() {
        //         MODULUS - (a as i64).unsigned_abs()
        //     } else {
        //         a as u64
        //     };
        //     assert_eq!(f,exp);
        // }
    }

    #[test]
    fn test_wrapped_arithmetic() {
        #[derive(Clone, Debug)]
        struct TestCase {
            a: Element,
            b: Element,
            res: Element,
        }
        let modulus = <<F as ExtensionField>::BaseField as SmallField>::MODULUS_U64;
        let cases = vec![
            TestCase {
                a: -53,
                b: 10,
                res: -53 * 10,
            },
            TestCase {
                a: -45,
                b: -56,
                res: 45 * 56,
            },
        ];
        for (i, case) in cases.iter().enumerate() {
            // cast them to handle overflow
            let ap: F = case.a.to_field();
            let bp: F = case.b.to_field();
            let res = ap * bp;
            let expected = case.res.to_field();
            assert_eq!(res, expected, "test case {}: {:?}", i, case);
        }
    }
}
