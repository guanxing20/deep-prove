use ff_ext::ExtensionField;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    Element,
    activation::{Activation, Relu},
    matrix::Matrix,
    vector_to_field_par, vector_to_field_par_into,
};

// The index of the step, starting from the input layer. (proving is done in the opposite flow)
pub type StepIdx = usize;

#[derive(Clone, Debug)]
pub enum Layer {
    // TODO: replace this with a Tensor based implementation
    Dense(Matrix<Element>),
    Activation(Activation),
}

impl Layer {
    /// Run the operation associated with that layer with the given input
    // TODO: move to tensor library : right now it works because we assume there is only Dense
    // layer which is matmul
    pub fn op(&self, input: &[Element]) -> Vec<Element> {
        match self {
            Layer::Dense(ref matrix) => matrix.matmul(input),
            Layer::Activation(activation) => activation.op(input),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Layer::Dense(ref matrix) => vec![matrix.nrows(), matrix.ncols()],
            Layer::Activation(Activation::Relu(_)) => Relu::shape(),
        }
    }
    pub fn to_string(&self) -> String {
        match self {
            Layer::Dense(ref matrix) => {
                format!("Dense: ({},{})", matrix.nrows(), matrix.ncols())
            }
            Layer::Activation(Activation::Relu(_)) => {
                format!("RELU: {}", 1 << Relu::num_vars())
            }
        }
    }
}

/// NOTE: this doesn't handle dynamism in the model with loops for example for LLMs where it
/// produces each token one by one.
#[derive(Clone, Debug)]
pub struct Model {
    layers: Vec<Layer>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            layers: Default::default(),
        }
    }
    pub fn add_layer(&mut self, l: Layer) {
        self.layers.push(l);
    }

    pub fn run<'a, E: ExtensionField>(&'a self, input: Vec<Element>) -> InferenceTrace<'a, E> {
        let mut trace = InferenceTrace::<Element>::new(input);
        for (id, layer) in self.layers() {
            let input = trace.last_input();
            let output = layer.op(input);
            let step = InferenceStep { layer, output, id };
            trace.push_step(step);
        }
        trace.to_field()
    }

    pub fn layers(&self) -> impl DoubleEndedIterator<Item = (StepIdx, &Layer)> {
        self.layers.iter().enumerate()
    }

    pub fn input_shape(&self) -> Vec<usize> {
        let Layer::Dense(mat) = &self.layers[0] else {
            panic!("layer is not starting with a dense layer?");
        };
        vec![mat.ncols()]
    }

    pub fn first_output_shape(&self) -> Vec<usize> {
        let Layer::Dense(mat) = &self.layers[0] else {
            panic!("layer is not starting with a dense layer?");
        };
        vec![mat.nrows()]
    }
    /// Prints to stdout
    pub fn describe(&self) {
        println!("MATRIX description:");
        for (idx, layer) in self.layers() {
            println!("\t- {}: {:?}", idx, layer.to_string());
        }
        println!("\n");
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

/// Keeps track of all input and outputs of each layer, with a reference to the layer.
pub struct InferenceTrace<'a, E> {
    steps: Vec<InferenceStep<'a, E>>,
    /// The initial input to the model
    input: Vec<E>,
}

impl<'a> InferenceTrace<'a, Element> {
    pub fn to_field<E: ExtensionField>(self) -> InferenceTrace<'a, E> {
        let input = vector_to_field_par_into(self.input);
        let field_steps = self
            .steps
            .par_iter()
            .map(|step| InferenceStep {
                id: step.id,
                layer: step.layer,
                output: vector_to_field_par(&step.output),
            })
            .collect::<Vec<_>>();
        InferenceTrace {
            steps: field_steps,
            input,
        }
    }
}

impl<'a, E> InferenceTrace<'a, E> {
    fn new(input: Vec<E>) -> Self {
        Self {
            steps: Default::default(),
            input,
        }
    }
    pub fn last_step(&self) -> &InferenceStep<'a, E> {
        self.steps
            .last()
            .expect("can't call last_step on empty inferece")
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

    /// Returns the final output of the whole trace
    pub fn final_output(&self) -> &[E] {
        &self
            .steps
            .last()
            .expect("can't call final_output on empty trace")
            .output
    }

    fn push_step(&mut self, step: InferenceStep<'a, E>) {
        self.steps.push(step);
    }

    /// Returns an iterator over (input, step) pairs
    pub fn iter(&self) -> InferenceTraceIterator<'_, 'a, E> {
        InferenceTraceIterator {
            trace: self,
            current_idx: 0,
            end_idx: self.steps.len(),
        }
    }
}

/// Iterator that yields (input, step) pairs for each inference step
pub struct InferenceTraceIterator<'t, 'a, E> {
    trace: &'t InferenceTrace<'a, E>,
    current_idx: usize,
    /// For double-ended iteration
    end_idx: usize,
}

impl<'t, 'a, E> Iterator for InferenceTraceIterator<'t, 'a, E> {
    type Item = (&'t [E], &'t InferenceStep<'a, E>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.end_idx {
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

impl<'t, 'a, E> DoubleEndedIterator for InferenceTraceIterator<'t, 'a, E> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.end_idx {
            return None;
        }

        self.end_idx -= 1;
        let step = &self.trace.steps[self.end_idx];
        let input = if self.end_idx == 0 {
            &self.trace.input
        } else {
            &self.trace.steps[self.end_idx - 1].output
        };

        Some((input, step))
    }
}

pub struct InferenceStep<'a, E> {
    pub id: StepIdx,
    /// Reference to the layer that produced this step
    pub layer: &'a Layer,
    /// Output produced by this layer
    pub output: Vec<E>,
}

#[cfg(test)]
pub(crate) mod test {
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;

    use crate::{
        Element,
        activation::{Activation, Relu},
        matrix::Matrix,
        model::Layer,
        testing::random_vector,
        vector_to_field_par,
    };

    use super::Model;

    type F = GoldilocksExt2;
    const SELECTOR_DENSE: usize = 0;
    const SELECTOR_RELU: usize = 1;
    const MOD_SELECTOR: usize = 2;

    impl Model {
        /// Returns a random model with specified number of dense layers and a matching input.
        pub fn random(num_dense_layers: usize) -> (Self, Vec<Element>) {
            let mut model = Model::new();
            let mut rng = thread_rng();
            let mut last_row = rng.gen_range(3..15);
            for selector in 0..num_dense_layers {
                if selector % MOD_SELECTOR == SELECTOR_DENSE {
                    // if true {
                    // last row becomes new column
                    let (nrows, ncols) = (rng.gen_range(3..15), last_row);
                    last_row = nrows;
                    let mat = Matrix::random((nrows, ncols)).pad_next_power_of_two();
                    model.add_layer(Layer::Dense(mat));
                } else if selector % MOD_SELECTOR == SELECTOR_RELU {
                    model.add_layer(Layer::Activation(Activation::Relu(Relu::new())));
                    // no need to change the `last_row` since RELU layer keeps the same shape
                    // of outputs
                } else {
                    panic!("random selection shouldn't be in that case");
                }
            }
            let input_dims = model.layers.first().unwrap().shape();
            // ncols since matrix2vector is summing over the columns
            let input = random_vector(input_dims[1]);
            (model, input)
        }
    }

    #[test]
    fn test_model_long() {
        let (model, input) = Model::random(3);
        model.run::<F>(input);
    }

    #[test]
    fn test_model_run() {
        let mat1 = Matrix::random((10, 11)).pad_next_power_of_two();
        let mat2 = Matrix::random((7, mat1.ncols())).pad_next_power_of_two();
        let input = random_vector::<Element>(mat1.ncols());
        let output1 = mat1.matmul(&input);
        let final_output = mat2.matmul(&output1);

        let mut model = Model::new();
        model.add_layer(Layer::Dense(mat1));
        model.add_layer(Layer::Dense(mat2.clone()));

        let trace = model.run::<F>(input.clone());
        assert_eq!(trace.steps.len(), 2);

        // Verify first step
        assert_eq!(trace.steps[0].output, vector_to_field_par(&output1));

        // Verify second step
        assert_eq!(trace.steps[1].output, vector_to_field_par(&final_output));
        let (nrow, _) = (mat2.nrows(), mat2.ncols());
        assert_eq!(final_output.len(), nrow);
    }

    #[test]
    fn test_inference_trace_iterator() {
        let mat1 = Matrix::random((10, 11)).pad_next_power_of_two();
        let relu1 = Activation::Relu(Relu);
        let mat2 = Matrix::random((7, mat1.ncols())).pad_next_power_of_two();
        let relu2 = Activation::Relu(Relu);
        let input = random_vector(mat1.ncols());

        let mut model = Model::new();
        model.add_layer(Layer::Dense(mat1));
        model.add_layer(Layer::Dense(mat2));

        let trace = model.run::<F>(input.clone());

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

    #[test]
    fn test_inference_trace_reverse_iterator() {
        let mat1 = Matrix::random((10, 11)).pad_next_power_of_two();
        let mat2 = Matrix::random((7, mat1.ncols())).pad_next_power_of_two();
        let input = random_vector(mat1.ncols());

        let mut model = Model::new();
        model.add_layer(Layer::Dense(mat1));
        model.add_layer(Layer::Dense(mat2));

        let trace = model.run::<F>(input.clone());

        // Test reverse iteration
        let mut rev_iter = trace.iter().rev();

        // Last step should come first in reverse
        let (last_input, last_step) = rev_iter.next().unwrap();
        assert_eq!(last_input, trace.steps[0].output);
        assert_eq!(last_step.output, trace.steps[1].output);

        // First step should come last in reverse
        let (first_input, first_step) = rev_iter.next().unwrap();
        assert_eq!(first_input, trace.input);
        assert_eq!(first_step.output, trace.steps[0].output);

        // Iterator should be exhausted
        assert!(rev_iter.next().is_none());
    }
}
