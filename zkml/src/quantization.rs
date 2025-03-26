//! Module that takes care of (re)quantizing
use std::ops::{Add, Div, Mul, Sub};
use derive_more::From;
use ff_ext::ExtensionField;
use goldilocks::SmallField;
use itertools::Itertools;
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::env;
use tracing::debug;

use crate::{Element, tensor::Tensor};

// Get BIT_LEN from environment variable or use default value
pub static BIT_LEN: Lazy<usize> = Lazy::new(|| {
    env::var("ZKML_BIT_LEN")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(8) // Default value if env var is not set or invalid
});

// These values depend on BIT_LEN and need to be computed at runtime
pub static MIN: Lazy<Element> = Lazy::new(|| -(1 << (*BIT_LEN - 1)));
pub static MAX: Lazy<Element> = Lazy::new(|| (1 << (*BIT_LEN - 1)) - 1);
pub static ZERO: Lazy<Element> = Lazy::new(|| 0);
pub static RANGE_BITLEN: Lazy<Element> = Lazy::new(|| (*MAX - *ZERO).ilog2() as Element);

/// Each scaling factor is of the form 2^k / 2^BIT_LEN. THis struct stores k.
#[derive(Debug,Clone, From,Copy)]
pub struct ScalingFactor(pub(crate) usize);
impl ScalingFactor  {
   /// Derives the right shift to apply to values to requantize them
   /// S_i = (a_i - b_i) / 2^Q = 2^k_i / 2^Q
   /// So S1 * S2 / S3 = 2^k1 * 2^k2 * 2^Q / [2^k3 * 2^Q * 2^Q]
   ///                 = 2^(k1 + k2 - k3 - Q)
   ///                 = 2^{-n} where n = k3 + Q - k1 - k2
   /// n is the number of bits to shift right
   pub fn shift(&self,s2: Self, s3: Self) -> usize {
        let twoq = (1 << *BIT_LEN) as f32;
        let fulls1 = self.0 as f32 / twoq;
        let fulls2 = s2.0 as f32 / twoq;
        let fulls3 = s3.0 as f32 / twoq;
        let full = fulls1 * fulls2 / fulls3;
        assert!(full >= 0.0 && full <= 1.0, "Full is not in the range [0, 1]. This should not happen.");
        println!("SHIFT DENSE s1={:?}, s2={:?},s3={:?} => shift={}", self,s2,s3, self.0 as i32 + s2.0 as i32 - s3.0  as i32 - *BIT_LEN as i32);
        let exp = self.0 as i32 + s2.0 as i32 - s3.0  as i32 - *BIT_LEN as i32;
        assert!(exp <= 0, "Shift exp {} is positive. This should not happen. (full={})", exp, full);
        exp.abs() as usize
   } 
}

impl Default for ScalingFactor {
    fn default() -> Self {
        // (a -b) / 2^Q : for scaling factor we only keep the numerator since denumerator is always the same.
        // in this case: max = 1, min = -1, (a-b) = 2, so k = 1
        Self(1)
    }
}

/// Trait used to quantize original floating point number to integer
pub trait Quantizer<Output> {
    fn from_f32_unsafe(e: &f32) -> Output;
    fn from_f32_unsafe_clamp(e: &f32, max_abs: f64) -> Output;
}

impl Quantizer<Element> for Element {
    fn from_f32_unsafe(e: &f32) -> Self {
        assert!(
            *e >= -1.0 && *e <= 1.0,
            "Input value must be between -1.0 and 1.0"
        );
        let scale = ScalingFactor::default().0 as f32/ (1 << *BIT_LEN) as f32;
        let zero_point = 0;

        // formula is q = round(r/S) + z
        let scaled = (*e / scale).round() as Element + zero_point;
        scaled as Element
    }

    fn from_f32_unsafe_clamp(e: &f32, max_abs_range: f64) -> Self {
        let e = *e as f64;
        assert!(
            max_abs_range > 0.0,
            "max_abs should be greater than zero. Domain range is between [-max_abs, max_abs]."
        );

        let scale = max_abs_range / (1 << *BIT_LEN) as f64;
        //let iscale = 1.0 / scale;
        //let piscale = (iscale as u32).next_power_of_two() as f64;
        //let fscale = 1.0 /piscale;
        let zero_point = 0;

        // formula is q = round(r/S) + z
        let scaled = (e / scale).round() as Element + zero_point;
        let scaled = scaled.clamp(*MIN, *MAX);

        //if e < -max_abs || e > max_abs {
        //    debug!(
        //        "Quantization: Value {} is out of [-{}, {}]. But quantized to {}.",
        //        e, max_abs, max_abs, scaled
        //    );
        //}
        scaled as Element
    }
}

pub(crate) trait Fieldizer<F> {
    fn to_field(&self) -> F;
}

impl<F: ExtensionField> Fieldizer<F> for Element {
    fn to_field(&self) -> F {
        if self.is_negative() {
            // Doing wrapped arithmetic : p-128 ... p-1 means negative number
            F::from(<F::BaseField as SmallField>::MODULUS_U64 - self.unsigned_abs() as u64)
        } else {
            // for positive and zero, it's just the number
            F::from(*self as u64)
        }
    }
}
pub(crate) trait IntoElement {
    fn into_element(&self) -> Element;
}

impl<F: ExtensionField> IntoElement for F {
    fn into_element(&self) -> Element {
        let e = self.to_canonical_u64_vec()[0] as Element;
        let modulus_half = <F::BaseField as SmallField>::MODULUS_U64 >> 1;
        // That means he's a positive number
        if *self == F::ZERO {
            0
        // we dont assume any bounds on the field elements, requant might happen at a later stage
        // so we assume the worst case
        } else if e <= modulus_half as Element {
            e
        } else {
            // That means he's a negative number - so take the diff with the modulus and recenter around 0
            let diff = <F::BaseField as SmallField>::MODULUS_U64 - e as u64;
            -(diff as Element)
        }
    }
}

impl<F: ExtensionField> Fieldizer<F> for u8 {
    fn to_field(&self) -> F {
        F::from(*self as u64)
    }
}

pub trait TensorFielder<F> {
    fn to_fields(self) -> Tensor<F>;
}

impl<F: ExtensionField, T> TensorFielder<F> for Tensor<T>
where
    T: Fieldizer<F>,
{
    fn to_fields(self) -> Tensor<F> {
        Tensor::new(
            self.get_shape(),
            self.get_data()
                .into_iter()
                .map(|i| i.to_field())
                .collect_vec(),
        )
    }
}

pub fn range_from_weight(weight: &Element) -> (Element, Element) {
    let min = if weight.is_negative() {
        weight * *MAX as Element
    } else {
        weight * *MIN as Element
    };
    let max = if weight.is_negative() {
        weight * *MIN as Element
    } else {
        weight * *MAX as Element
    };
    (min, max)
}

#[cfg(test)]
mod test {
    use crate::quantization::Fieldizer;

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

#[cfg(test)]
mod tests {
    use super::*;

    type F = goldilocks::GoldilocksExt2;
    #[test]
    fn test_element_field_roundtrip() {
        // Also test a few specific values explicitly
        let test_values = [*MIN, -100, -50, -1, 0, 1, 50, 100, *MAX];
        for &val in &test_values {
            let field_val: F = val.to_field();
            let roundtrip = field_val.into_element();

            assert_eq!(
                val, roundtrip,
                "Element {} did not roundtrip correctly (got {})",
                val, roundtrip
            );
        }
    }
    use std::ops::{Mul,Add,Div};

    #[test]
    fn test_scaling_factor() {
        let s1 :ScalingFactor= ScalingFactor::from(2);
        let s2 :ScalingFactor= ScalingFactor::from(2);
        let s3 = ScalingFactor::from(4);
        let shift = s1.shift(s2, s3);
        assert_eq!(shift, s3.0 + *BIT_LEN - s1.0 - s2.0);
    }
}
