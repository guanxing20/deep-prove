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
        let mut trace = InferenceTrace::new(input);
        for layer in &self.layers {
            let input = trace.last_input();
            let output = layer.eval(input);
            let step = InferenceStep { layer, output };
            trace.push_step(step);
        }
        trace
    }
}

/// Keeps track of all input and outputs of each layer, with a reference to the layer.
pub struct InferenceTrace<'a, E> {
    steps: Vec<InferenceStep<'a, E>>,
    /// The initial input to the model
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

    fn push_step(&mut self, step: InferenceStep<'a, E>) {
        self.steps.push(step);
    }

    /// Returns an iterator over (input, step) pairs
    pub fn iter(&self) -> InferenceTraceIterator<'_, 'a, E> {
        InferenceTraceIterator {
            trace: self,
            current_idx: 0,
        }
    }
}

/// Iterator that yields (input, step) pairs for each inference step
pub struct InferenceTraceIterator<'t, 'a, E> {
    trace: &'t InferenceTrace<'a, E>,
    current_idx: usize,
}

impl<'t, 'a, E> Iterator for InferenceTraceIterator<'t, 'a, E> {
    type Item = (&'t [E], &'t InferenceStep<'a, E>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.trace.steps.len() {
            return None;
        }

        let step = &self.trace.steps[self.current_idx];
        let input = if self.current_idx == 0 {
            &self.trace.input
        } else {
            &self.trace.steps[self.current_idx - 1].output
        };

        self.current_idx += 1;
        Some((input, step))
    }
}

pub struct InferenceStep<'a, E> {
    /// Reference to the layer that produced this step
    pub layer: &'a Layer<E>,
    /// Output produced by this layer
    pub output: Vec<E>,
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
        let mat1 = Matrix::<E>::random((10, 11)).pad_next_power_of_two();
        let mat2 = Matrix::<E>::random((7, mat1.ncols())).pad_next_power_of_two();
        let input = (0..mat1.ncols())
            .map(|_| E::random(&mut thread_rng()))
            .collect_vec();
        let output1 = mat1.matmul(&input);
        let final_output = mat2.matmul(&output1);

        let mut model = Model::<E>::new();
        model.add_layer(Layer::Dense(mat1));
        model.add_layer(Layer::Dense(mat2));

        let trace = model.run(input.clone());
        assert_eq!(trace.steps.len(), 2);

        // Verify first step
        assert_eq!(trace.steps[0].output, output1);

        // Verify second step
        assert_eq!(trace.steps[1].output, final_output);
    }

    #[test]
    fn test_inference_trace_iterator() {
        let mat1 = Matrix::<E>::random((10, 11)).pad_next_power_of_two();
        let mat2 = Matrix::<E>::random((7, mat1.ncols())).pad_next_power_of_two();
        let input = (0..mat1.ncols())
            .map(|_| E::random(&mut thread_rng()))
            .collect_vec();

        let mut model = Model::<E>::new();
        model.add_layer(Layer::Dense(mat1));
        model.add_layer(Layer::Dense(mat2));

        let trace = model.run(input.clone());

        // Verify iterator yields correct input/output pairs
        let mut iter = trace.iter();

        // First step should have original input
        let (first_input, first_step) = iter.next().unwrap();
        assert_eq!(first_input, trace.input);
        assert_eq!(first_step.output, trace.steps[0].output);

        // Second step should have first step's output as input
        let (second_input, second_step) = iter.next().unwrap();
        assert_eq!(second_input, trace.steps[0].output);
        assert_eq!(second_step.output, trace.steps[1].output);

        // Iterator should be exhausted
        assert!(iter.next().is_none());
    }
}
