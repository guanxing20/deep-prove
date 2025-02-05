use std::{default, sync::Arc};

use ff_ext::ExtensionField;

use crate::matrix::Matrix;

pub enum Layer<E> {
    // TODO: replace this with a Tensor based implementation
    Dense(Matrix<E>),
}

impl<E: ExtensionField> Layer<E> {
    pub fn dim(&self) -> (usize, usize) {
        match self {
            Layer::Dense(ref matrix) => (matrix.nrows(), matrix.ncols()),
        }
    }
    // TODO: move to tensor library : right now it works because we assume there is only Dense
    // layer which is matmul
    pub fn eval(&self, input: &[E]) -> Vec<E> {
        match self {
            Layer::Dense(ref matrix) => matrix.matmul(input),
        }
    }
}

/// NOTE: this doesn't handle dynamism in the model with loops for example for LLMs where it
/// produces each token one by one.
pub struct Model<E> {
    layers: Vec<Layer<E>>,
}

impl<E: ExtensionField> Model<E> {
    pub fn new() -> Self {
        Self {
            layers: Default::default(),
        }
    }
    pub fn add_layer(&mut self, l: Layer<E>) {
        self.layers.push(l);
    }
    pub fn run<'a>(&'a self, input: Vec<E>) -> InferenceTrace<'a, E> {
        self.layers
            .iter()
            .fold(InferenceTrace::new(input), |mut trace, layer| {
                let output = layer.eval(trace.last_input());
                let step = InferenceStep { layer, output };
                trace.push_step(step);
                trace
            })
    }
}

/// Keeps track of all input and outputs of each layer, with a reference to the layer.
pub struct InferenceTrace<'a, E> {
    steps: Vec<InferenceStep<'a, E>>,
    input: Vec<E>,
}

impl<'a, E> InferenceTrace<'a, E> {
    fn new(input: Vec<E>) -> Self {
        Self {
            steps: Default::default(),
            input,
        }
    }

    /// Useful when building the trace. The next input is either the first input or the last
    /// output.
    fn last_input(&self) -> &[E] {
        if self.steps.is_empty() {
            &self.input
        } else {
            // safe unwrap since it's not empty
            &self.steps.last().unwrap().output
        }
    }

    /// Once a layer have been evaluated, add it to the trace.
    fn push_step(&mut self, step: InferenceStep<'a, E>) {
        self.steps.push(step);
    }
}

pub struct InferenceStep<'a, E> {
    layer: &'a Layer<E>,
    // only need to keep output since input is known from the execution context
    output: Vec<E>,
}

#[cfg(test)]
mod test {
    use ark_std::rand::thread_rng;
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;

    use crate::{matrix::Matrix, model::Layer};

    use super::Model;
    use ff_ext::ff::Field;

    type E = GoldilocksExt2;
    #[test]
    fn test_model_run() {
        // 1. define 2 matrices and one input
        // 2. manually compute the first output and final output
        // 3. create the model and run it
        // 4. compare outputs
        //
        let mat1 = Matrix::<E>::random((10, 11)).pad_next_power_of_two();
        let mat2 = Matrix::<E>::random((7, mat1.ncols())).pad_next_power_of_two();
        let input = (0..mat1.ncols())
            .map(|_| E::random(&mut thread_rng()))
            .collect_vec();
        let output1 = mat1.matmul(&input);
        assert_eq!(
            output1.len(),
            mat2.ncols(),
            "({},{})",
            mat2.nrows(),
            mat2.ncols()
        );
        let final_output = mat2.matmul(&output1);
        let mut model = Model::<E>::new();
        model.add_layer(Layer::Dense(mat1.clone()));
        model.add_layer(Layer::Dense(mat2.clone()));
        let trace = model.run(input.clone());
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.input, input);
        assert_eq!(trace.steps[0].output, output1);
        assert_eq!(trace.steps[1].output, final_output);
    }
}
