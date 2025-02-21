use std::fmt::Write;

use anyhow::ensure;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::mle::DenseMultilinearExtension;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::{
    Element,
    quantization::{Fieldizer, QuantInteger},
    testing::random_vector,
    to_bit_sequence_le,
};

#[derive(Clone, Debug, Default)]
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
    pub fn shape(&self) -> Vec<usize> {
        vec![self.nrows(), self.ncols()]
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
            .map(|e| e.to_field())
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
        let coeffs = random_vector::<QuantInteger>(rows * cols)
            .into_par_iter()
            .map(|r| r as Element)
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
        writeln!(out, "Matrix({},{})", self.nrows(), self.ncols()).expect("...");
        for (i, row) in self.coeffs.iter().enumerate() {
            writeln!(out, "{}: {:?}", i, row).expect("..");
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

    /// Reshapes the matrix to have at least the specified dimensions while preserving all data.
    pub fn reshape_to_fit_inplace(&mut self, new_rows: usize, new_cols: usize) {
        // Ensure we never lose information by requiring the new dimensions to be at least
        // as large as the original ones
        assert!(
            new_rows >= self.nrows(),
            "Cannot shrink matrix rows from {} to {} - would lose information",
            self.nrows(),
            new_rows
        );
        assert!(
            new_cols >= self.ncols(),
            "Cannot shrink matrix columns from {} to {} - would lose information",
            self.ncols(),
            new_cols
        );

        // Create a new matrix with expanded dimensions
        let new_coeffs: Vec<Vec<Element>> = (0..new_rows)
            .map(|i| {
                (0..new_cols)
                    .map(|j| {
                        if i < self.nrows() && j < self.ncols() {
                            self.coeffs[i][j]
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect();

        self.dim = (new_rows, new_cols);
        self.coeffs = new_coeffs;
    }
}

#[cfg(test)]
mod test {
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::{
        mle::{IntoMLE, MultilinearExtension},
        virtual_poly::VirtualPolynomial,
    };
    use sumcheck::structs::IOPProverState;

    use crate::{
        Element,
        quantization::{QuantInteger, VecFielder},
        testing::{random_bool_vector, random_vector},
    };

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
        let (n_rows, n_cols): (usize, usize) = (10, 10);
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

    type F = GoldilocksExt2;
    use crate::default_transcript;
    use ff::Field;
    #[test]
    fn test_matrix_proving_sequential() {
        let nrows = 8;
        let m = Matrix::random((nrows, nrows)).pad_next_power_of_two();
        let mmle = m.to_mle();
        let input = random_vector::<QuantInteger>(nrows)
            .into_iter()
            .map(|i| i as Element)
            .collect_vec();
        println!("{}", m.fmt_integer());
        let output = m.matmul(&input);
        let point1 = random_bool_vector(nrows.ilog2() as usize);
        println!("point1: {:?}", point1);
        let inputf: Vec<F> = input.as_slice().to_fields();
        println!("input: {:?}", input);
        println!("inputF: {:?}", inputf);
        let outputf: Vec<F> = output.as_slice().to_fields();
        let computed_eval1 = outputf.into_mle().evaluate(&point1);
        let flatten_mat1 = mmle.fix_high_variables(&point1);
        // y(r) = SUM_i m(r,i) x(i)
        let full_poly = vec![flatten_mat1.clone().into(), inputf.into_mle().into()];
        let mut vp = VirtualPolynomial::new(flatten_mat1.num_vars());
        vp.add_mle_list(full_poly, F::ONE);
        #[allow(deprecated)]
        let (proof, state) =
            IOPProverState::<F>::prove_parallel(vp.clone(), &mut default_transcript());
        let (p2, s2) = IOPProverState::prove_batch_polys(1, vec![vp], &mut default_transcript());
        let given_eval1 = proof.extract_sum();
        assert_eq!(p2.extract_sum(), proof.extract_sum());
        assert_eq!(computed_eval1, given_eval1);
    }
}
