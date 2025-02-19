use num_traits::Float; // Import the Float trait
use std::cmp::Ord; // For clamping

// https://pytorch.org/blog/quantization-in-practice/
// Quantize: q = round(r / S) + Z
// Constrain T to be a floating-point type (f32 or f64)
fn affine_quantize<T>(r: T, scale: T, zero_point: i64, min_q: i64, max_q: i64) -> i64
where
    T: Float,
{
    let q = (r / scale).round().to_i64().unwrap() + zero_point;
    q.clamp(min_q, max_q)
}

// Dequantize: r = S * (q - Z)
fn affine_dequantize(q: i64, scale: f32, zero_point: i64) -> f32 {
    scale * (q - zero_point) as f32
}

mod test {
    use super::*;

    #[test]
    fn test_utils_quantize() {
        let zero_point = 128;
        let min_q = 0;
        let max_q = 255;

        let scale = 2.0 / 255.0;
        let r = -1.0 as f32;
        assert!(affine_quantize(r, scale, zero_point, min_q, max_q) == 1);

        let r = 0.0 as f32;
        assert!(affine_quantize(r, scale, zero_point, min_q, max_q) == 128);

        let r = 1.0 as f32;
        assert!(affine_quantize(r, scale, zero_point, min_q, max_q) == 255);

        let scale = 2.0 / 255.0;
        let r = -1.0 as f64;
        assert!(affine_quantize(r, scale, zero_point, min_q, max_q) == 0);

        let r = 0.0 as f64;
        assert!(affine_quantize(r, scale, zero_point, min_q, max_q) == 128);

        let r = 1.0 as f64;
        assert!(affine_quantize(r, scale, zero_point, min_q, max_q) == 255);
    }
}
