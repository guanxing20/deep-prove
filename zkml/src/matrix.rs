use std::fmt::Write;

use anyhow::ensure;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::mle::DenseMultilinearExtension;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::{Element, testing::random_vector, to_bit_sequence_le};

#[derive(Clone, Debug)]
pub struct Matrix<E> {
    dim: (usize, usize),
    // dimension is [n_rows,n_cols]
    coeffs: Vec<Vec<E>>,
}

impl Matrix<Element> {
    pub fn from_coeffs(coeffs: Vec<Vec<Element>>) -> anyhow::Result<Self> {
        let n_rows = coeffs.len();
        let n_cols = coeffs.first().expect("at least one row in a matrix").len();
        for row in &coeffs {
            ensure!(n_cols == row.len());
        }
        Ok(Self {
            dim: (n_rows, n_cols),
            coeffs,
        })
    }
    /// From a given row and a given column, return the vector of field elements in the right
    /// format to evaluate the MLE.
    /// little endian so we need to read cols before rows
    pub fn position_to_boolean<F: ExtensionField>(&self, row: usize, col: usize) -> Vec<F> {
        self.col_to_boolean(col)
            .chain(self.row_to_boolean(row))
            .collect_vec()
    }
    /// Returns the boolean iterator indicating the given row in the right endianness to be
    /// evaluated by an MLE
    pub fn row_to_boolean<F: ExtensionField>(&self, row: usize) -> impl Iterator<Item = F> {
        let (nvars_rows, _) = self.num_vars();
        to_bit_sequence_le(row, nvars_rows).map(|b| F::from(b as u64))
    }
    /// Returns the boolean iterator indicating the given row in the right endianness to be
    /// evaluated by an MLE
    pub fn col_to_boolean<F: ExtensionField>(&self, col: usize) -> impl Iterator<Item = F> {
        let (_, nvars_col) = self.num_vars();
        to_bit_sequence_le(col, nvars_col).map(|b| F::from(b as u64))
    }

    /// Returns the number of boolean variables needed to address any row, and any columns
    pub fn num_vars(&self) -> (usize, usize) {
        (self.nrows().ilog2() as usize, self.ncols().ilog2() as usize)
    }

    /// Returns a MLE of the matrix that can be evaluated.
    pub fn to_mle<F: ExtensionField>(&self) -> DenseMultilinearExtension<F> {
        assert!(
            self.nrows().is_power_of_two(),
            "number of rows {} is not a power of two",
            self.nrows()
        );
        assert!(
            self.ncols().is_power_of_two(),
            "number of columns {} is not a power of two",
            self.ncols()
        );
        // N variable to address 2^N rows and M variables to address 2^M columns
        let num_vars = self.nrows().ilog2() + self.ncols().ilog2();
        DenseMultilinearExtension::from_evaluations_ext_vec(num_vars as usize, self.evals())
    }

    /// Returns the evaluation point, in order for (row,col) addressing
    pub fn evals<F: ExtensionField>(&self) -> Vec<F> {
        self.coeffs
            .par_iter()
            .flatten()
            .map(|e| F::from(*e as u64))
            .collect()
    }

    pub fn pad_next_power_of_two(mut self) -> Self {
        // assume the matrix is already well formed and there is always n_rows and n_cols
        // this is because we control the creation of the matrix in the first place
        let new_rows = if self.nrows().is_power_of_two() {
            self.nrows()
        } else {
            self.nrows().next_power_of_two()
        };
        let new_cols = if self.ncols().is_power_of_two() {
            self.ncols()
        } else {
            self.ncols().next_power_of_two()
        };
        self.dim = (new_rows, new_cols);
        // resize each row
        for row in self.coeffs.iter_mut() {
            if row.len() != new_cols {
                row.resize(new_cols, 0);
            }
        }
        // resize the number of rows
        if self.coeffs.len() != new_rows {
            self.coeffs.resize(new_rows, vec![0; new_cols]);
        }
        self
    }
    /// Creates a random matrix with a given number of rows and cols.
    /// NOTE: doesn't take a rng as argument because to generate it in parallel it needs be sync +
    /// sync which is not true for basic rng core.
    pub fn random((rows, cols): (usize, usize)) -> Self {
        let coeffs = random_vector(rows * cols)
            .into_par_iter()
            .chunks(cols)
            .collect();
        Self {
            dim: (rows, cols),
            coeffs,
        }
    }
    pub fn nrows(&self) -> usize {
        self.dim.0
    }
    pub fn ncols(&self) -> usize {
        self.dim.1
    }

    pub fn fmt_integer(&self) -> String {
        let mut out = String::new();
        write!(out, "Matrix({},{})\n", self.nrows(), self.ncols()).expect("...");
        for (i, row) in self.coeffs.iter().enumerate() {
            write!(out, "{}: {:?}\n", i, row).expect("..");
        }
        out
    }

    /// Performs vector matrix multiplication in a school book naive way.
    /// TODO: actually getting the result should be done via proper tensor-like libraries like
    /// candle that can handle this algo much faster
    pub fn matmul(&self, vec: &[Element]) -> Vec<Element> {
        self.coeffs
            .par_iter()
            .map(|row| {
                // check the number of columns correspond to the length of the vector
                assert_eq!(row.len(), vec.len());
                // dot product
                row.clone()
                    .into_iter()
                    .zip(vec.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }

    // Computes returns the transpose of the current matrix
    pub fn transpose(&self) -> Matrix<Element> {
        let (rows, cols) = self.dim;
        let mut transposed_coeffs = vec![vec![self.coeffs[0][0].clone(); rows]; cols];

        for i in 0..rows {
            for j in 0..cols {
                transposed_coeffs[j][i] = self.coeffs[i][j].clone();
            }
        }

        Matrix {
            dim: (cols, rows),
            coeffs: transposed_coeffs,
        }
    }
}

#[cfg(test)]
mod test {
    use ark_std::rand::{Rng, thread_rng};
    use ff_ext::ExtensionField;
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::MultilinearExtension;

    use crate::Element;

    use super::Matrix;

    impl Matrix<Element> {
        pub fn assert_structure(&self, (n_rows, n_cols): (usize, usize)) {
            assert_eq!(self.dim.0, n_rows);
            assert_eq!(self.dim.1, n_cols);
            assert_eq!(n_rows, self.coeffs.len());
            for row in &self.coeffs {
                assert_eq!(n_cols, row.len());
            }
        }
        pub fn get(&self, i: usize, j: usize) -> Element {
            self.coeffs[i][j]
        }
        pub fn random_eval_point(&self) -> Vec<E> {
            let mut rng = thread_rng();
            let r = rng.gen_range(0..self.nrows());
            let c = rng.gen_range(0..self.ncols());
            self.position_to_boolean(r, c)
        }
    }

    type E = GoldilocksExt2;

    #[test]
    fn test_matrix_matmul() {
        let mat = vec![vec![1, 2], vec![3, 4]];
        let x = vec![5, 6];
        let out = vec![17, 39];
        let mat = Matrix::<Element>::from_coeffs(mat).unwrap();
        let res = mat.matmul(&x);
        assert_eq!(out, res);
    }

    #[test]
    fn test_matrix_mle() {
        let mat = Matrix::<Element>::random((3, 5)).pad_next_power_of_two();
        println!("matrix: {}", mat.fmt_integer());
        let mut mle = mat.clone().to_mle::<E>();
        let (chosen_row, chosen_col) = (
            thread_rng().gen_range(0..mat.dim.0),
            thread_rng().gen_range(0..mat.dim.1),
        );
        let elem = mat.get(chosen_row, chosen_col);
        println!("(x,y) = ({},{}) ==> {:?}", chosen_row, chosen_col, elem);
        let inputs = mat.position_to_boolean(chosen_row, chosen_col);
        let output = mle.evaluate(&inputs);
        assert_eq!(E::from(elem as u64), output);

        // now try to address one at a time, and starting by the row, which is the opposite order
        // of the boolean variables expected by the MLE API, given it's expecting in LE format.
        let row_input = mat.row_to_boolean(chosen_row);
        mle.fix_high_variables_in_place(&row_input.collect_vec());
        let col_input = mat.col_to_boolean(chosen_col);
        let output = mle.evaluate(&col_input.collect_vec());
        assert_eq!(E::from(elem as u64), output);
    }

    #[test]
    fn test_matrix_random() {
        let (n_rows, n_cols) = (10, 10);
        let mat = Matrix::<Element>::random((n_rows, n_cols));
        mat.assert_structure((n_rows, n_cols));
    }

    #[test]
    fn test_matrix_next_power_of_two() {
        let (n_rows, n_cols) = (10, 10);
        let mat = Matrix::<Element>::random((n_rows, n_cols));
        let (new_rows, new_cols) = (n_rows.next_power_of_two(), n_cols.next_power_of_two());
        let new_mat = mat.pad_next_power_of_two();
        new_mat.assert_structure((new_rows, new_cols));
    }

    impl Matrix<Element> {
        pub fn is_equal(&self, other: &Self) -> bool {
            if self.dim != other.dim {
                return false;
            }

            self.coeffs == other.coeffs
        }
    }
    #[test]
    fn test_matrix_transpose() {
        let mat = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let mat = Matrix::<Element>::from_coeffs(mat).unwrap();

        let trans_mat = vec![vec![1, 3, 5], vec![2, 4, 6]];
        let trans_mat = Matrix::<Element>::from_coeffs(trans_mat).unwrap();
        let res = mat.transpose();
        let result = trans_mat.is_equal(&res);
        assert!(result == true);
    }
}
