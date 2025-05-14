
#[cfg(test)]
pub(crate) mod test {
    use crate::{
        ScalingFactor,
        layers::{
            Layer,
            activation::{Activation, Relu},
            convolution::{Convolution, SchoolBookConv},
            dense::Dense,
            pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling},
            provable::{ProvableModel, evaluate_layer},
            requant::Requant,
        },
        padding::PaddingMode,
        quantization,
        testing::{random_bool_vector, random_vector},
    };
    use anyhow::Result;
    use ark_std::rand::{Rng, RngCore, thread_rng};
    use ff_ext::ExtensionField;
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::{
        mle::{IntoMLE, MultilinearExtension},
        virtual_poly::VirtualPolynomial,
    };
    use sumcheck::structs::{IOPProverState, IOPVerifierState};

    use crate::{Element, default_transcript, quantization::TensorFielder, tensor::Tensor};

    type F = GoldilocksExt2;
    const SELECTOR_DENSE: usize = 0;
    const SELECTOR_RELU: usize = 1;
    const SELECTOR_POOLING: usize = 2;
    const MOD_SELECTOR: usize = 2;

    impl ProvableModel<Element> {
        pub fn random(num_dense_layers: usize) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut rng = thread_rng();
            Self::random_with_rng(num_dense_layers, &mut rng)
        }
        /// Returns a random model with specified number of dense layers and a matching input.
        /// Note that currently everything is considered padded, e.g. unpadded_shape = padded_shape
        pub fn random_with_rng<R: RngCore>(
            num_dense_layers: usize,
            rng: &mut R,
        ) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut last_row: usize = rng.gen_range(3..15);
            let mut model = Self::new_from_input_shapes(
                vec![vec![last_row.next_power_of_two()]],
                PaddingMode::NoPadding,
            );
            let mut last_node_id = None;
            for selector in 0..num_dense_layers {
                if selector % MOD_SELECTOR == SELECTOR_DENSE {
                    // if true {
                    // last row becomes new column
                    let (nrows, ncols): (usize, usize) = (rng.gen_range(3..15), last_row);
                    last_row = nrows;
                    let dense =
                        Dense::random(vec![nrows.next_power_of_two(), ncols.next_power_of_two()]);
                    // Figure out the requant information such that output is still within range
                    let (min_output_range, max_output_range) =
                        dense.output_range(*quantization::MIN, *quantization::MAX);
                    let output_scaling_factor = ScalingFactor::from_scale(
                        ((max_output_range - min_output_range) as f64
                            / (*quantization::MAX - *quantization::MIN) as f64)
                            as f32,
                        None,
                    );
                    let input_scaling_factor = ScalingFactor::from_scale(1.0, None);
                    let max_model = dense.matrix.max_value().max(dense.bias.max_value()) as f32;
                    let model_scaling_factor = ScalingFactor::from_absolute_max(max_model, None);
                    let shift =
                        input_scaling_factor.shift(&model_scaling_factor, &output_scaling_factor);
                    let requant = Requant::new(min_output_range as usize, shift);
                    last_node_id =
                        Some(model.add_consecutive_layer(Layer::Dense(dense), last_node_id)?);
                    last_node_id =
                        Some(model.add_consecutive_layer(Layer::Requant(requant), last_node_id)?);
                } else if selector % MOD_SELECTOR == SELECTOR_RELU {
                    last_node_id = Some(model.add_consecutive_layer(
                        Layer::Activation(Activation::Relu(Relu::new())),
                        last_node_id,
                    )?);
                    // no need to change the `last_row` since RELU layer keeps the same shape
                    // of outputs
                } else if selector % MOD_SELECTOR == SELECTOR_POOLING {
                    // Currently unreachable until Model is updated to work with higher dimensional tensors
                    // TODO: Implement higher dimensional tensor functionality.
                    last_node_id = Some(model.add_consecutive_layer(
                        Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default())),
                        last_node_id,
                    )?);
                    last_row -= MAXPOOL2D_KERNEL_SIZE - 1;
                } else {
                    panic!("random selection shouldn't be in that case");
                }
            }
            model.route_output(None).unwrap();
            let inputs = model
                .input_shapes()
                .iter()
                .map(|shape| Tensor::random(shape))
                .collect();
            Ok((model, inputs))
        }

        /// Returns a model that only contains pooling and relu layers.
        /// The output [`Model`] will contain `num_layers` [`Maxpool2D`] layers and a [`Dense`] layer as well.
        pub fn random_pooling(num_layers: usize) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut rng = thread_rng();
            // Since Maxpool reduces the size of the output based on the kernel size and the stride we need to ensure that
            // Our starting input size is large enough for the number of layers.

            // If maxpool input matrix has dimensions w x h then output has width and height
            // out_w = (w - kernel_size) / stride + 1
            // out_h = (h - kenrel_size) / stride + 1
            // Hence to make sure we have a large enough tensor for the last step
            // we need to have that w_first > 2^{num_layers + 1} + 2^{num_layers}
            // and likewise for h_first.

            let minimum_initial_size = (1 << num_layers) * (3usize);

            let mut input_shape = (0..3)
                .map(|i| {
                    if i < 1 {
                        rng.gen_range(1..5usize).next_power_of_two()
                    } else {
                        (minimum_initial_size + rng.gen_range(1..4usize)).next_power_of_two()
                    }
                })
                .collect::<Vec<usize>>();

            let mut model = ProvableModel::new_from_input_shapes(
                vec![input_shape.clone()],
                PaddingMode::NoPadding,
            );

            let inputs = model
                .input_shapes()
                .iter()
                .map(|shape| Tensor::random(shape))
                .collect();

            let info = Maxpool2D::default();
            let mut last_node_id = None;
            for _ in 0..num_layers {
                input_shape
                    .iter_mut()
                    .skip(1)
                    .for_each(|dim| *dim = (*dim - info.kernel_size) / info.stride + 1);
                last_node_id = Some(model.add_consecutive_layer(
                    Layer::Pooling(Pooling::Maxpool2D(info)),
                    last_node_id,
                )?);
            }

            let (nrows, ncols): (usize, usize) =
                (rng.gen_range(3..15), input_shape.iter().product::<usize>());

            model.add_consecutive_layer(
                Layer::Dense(Dense::random(vec![
                    nrows.next_power_of_two(),
                    ncols.next_power_of_two(),
                ])),
                last_node_id,
            )?;

            model.route_output(None)?;

            Ok((model, inputs))
        }
    }

    #[test]
    fn test_model_long() {
        let (model, input) = ProvableModel::random(3).unwrap();
        model.run::<F>(&input).unwrap();
    }

    pub fn check_tensor_consistency_field<E: ExtensionField>(
        real_tensor: Tensor<E>,
        padded_tensor: Tensor<E>,
    ) {
        let n_x = padded_tensor.shape[1];
        for i in 0..real_tensor.shape[0] {
            for j in 0..real_tensor.shape[1] {
                for k in 0..real_tensor.shape[1] {
                    // if(real_tensor.data[i*real_tensor.shape[1]*real_tensor.shape[1]+j*real_tensor.shape[1]+k] > 0){
                    assert!(
                        real_tensor.data[i * real_tensor.shape[1] * real_tensor.shape[1]
                            + j * real_tensor.shape[1]
                            + k]
                            == padded_tensor.data[i * n_x * n_x + j * n_x + k],
                        "Error in tensor consistency"
                    );
                    //}else{
                    //   assert!(-E::from(-real_tensor.data[i*real_tensor.shape[1]*real_tensor.shape[1]+j*real_tensor.shape[1]+k] as u64) == E::from(padded_tensor.data[i*n_x*n_x + j*n_x + k] as u64) ,"Error in tensor consistency");
                    //}
                }

                // assert!(real_tensor.data[i*real_tensor.shape[1]*real_tensor.shape[1]+j ] == padded_tensor.data[i*n_x*n_x + j],"Error in tensor consistency");
            }
        }
    }

    fn random_vector_quant(n: usize) -> Vec<Element> {
        // vec![thread_rng().gen_range(-128..128); n]
        random_vector(n)
    }

    #[test]
    fn test_cnn() {
        let mut in_dimensions: Vec<Vec<usize>> =
            vec![vec![1, 32, 32], vec![16, 29, 29], vec![4, 26, 26]];

        for i in 0..in_dimensions.len() {
            for j in 0..in_dimensions[0].len() {
                in_dimensions[i][j] = (in_dimensions[i][j]).next_power_of_two();
            }
        }
        // println!("in_dimensions: {:?}", in_dimensions);
        let w1 = random_vector_quant(16 * 16);
        let w2 = random_vector_quant(16 * 4 * 16);
        let w3 = random_vector_quant(16 * 8);

        let shape1 = vec![1 << 4, 1 << 0, 1 << 2, 1 << 2]; // [16, 1, 4, 4]
        let shape2 = vec![1 << 2, 1 << 4, 1 << 2, 1 << 2]; // [4, 16, 4, 4]
        let shape3 = vec![1 << 1, 1 << 2, 1 << 2, 1 << 2]; // [2, 4, 4, 4]
        let bias1: Tensor<Element> = Tensor::zeros(vec![shape1[0]]);
        let bias2: Tensor<Element> = Tensor::zeros(vec![shape2[0]]);
        let bias3: Tensor<Element> = Tensor::zeros(vec![shape3[0]]);

        let trad_conv1: Tensor<Element> = Tensor::new(shape1.clone(), w1.clone());
        let trad_conv2: Tensor<i128> = Tensor::new(shape2.clone(), w2.clone());
        let trad_conv3: Tensor<i128> = Tensor::new(shape3.clone(), w3.clone());

        let input_shape = vec![1, 32, 32];

        let mut model =
            ProvableModel::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::Padding);
        let input = Tensor::random(&model.input_shapes()[0]);
        let first_id = model
            .add_consecutive_layer(
                Layer::Convolution(
                    Convolution::new(trad_conv1.clone(), bias1.clone())
                        .into_padded_and_ffted(&in_dimensions[0]),
                ),
                None,
            )
            .unwrap();
        let second_id = model
            .add_consecutive_layer(
                Layer::Convolution(
                    Convolution::new(trad_conv2.clone(), bias2.clone())
                        .into_padded_and_ffted(&in_dimensions[1]),
                ),
                Some(first_id),
            )
            .unwrap();
        let _third_id = model
            .add_consecutive_layer(
                Layer::Convolution(
                    Convolution::new(trad_conv3.clone(), bias3.clone())
                        .into_padded_and_ffted(&in_dimensions[2]),
                ),
                Some(second_id),
            )
            .unwrap();
        model.route_output(None).unwrap();

        // END TEST
        let trace = model.run::<F>(&vec![input.clone()]).unwrap();

        let mut model2 =
            ProvableModel::new_from_input_shapes(vec![input_shape], PaddingMode::NoPadding);
        let first_id = model2
            .add_consecutive_layer(
                Layer::SchoolBookConvolution(SchoolBookConv(Convolution::new(trad_conv1, bias1))),
                None,
            )
            .unwrap();
        let second_id = model2
            .add_consecutive_layer(
                Layer::SchoolBookConvolution(SchoolBookConv(Convolution::new(trad_conv2, bias2))),
                Some(first_id),
            )
            .unwrap();
        let _third_id = model2
            .add_consecutive_layer(
                Layer::SchoolBookConvolution(SchoolBookConv(Convolution::new(trad_conv3, bias3))),
                Some(second_id),
            )
            .unwrap();
        model2.route_output(None).unwrap();
        let trace2 = model.run::<F>(&vec![input]).unwrap();

        check_tensor_consistency_field::<GoldilocksExt2>(
            trace2.outputs().unwrap()[0].to_fields(),
            trace.outputs().unwrap()[0].to_fields(),
        );
    }

    #[test]
    fn test_conv_maxpool() {
        let input_shape = vec![3usize, 32, 32];
        let shape1 = vec![6, 3, 5, 5];
        let filter = Tensor::random(&shape1);
        let bias1 = Tensor::random(&vec![shape1[0]]);

        let mut model =
            ProvableModel::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::Padding);
        let conv_layer = model
            .add_consecutive_layer(
                Layer::Convolution(
                    Convolution::new(filter.clone(), bias1.clone())
                        .into_padded_and_ffted(&input_shape),
                ),
                None,
            )
            .unwrap();
        let _pool_layer = model
            .add_consecutive_layer(
                Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default())),
                Some(conv_layer),
            )
            .unwrap();
        model.route_output(None).unwrap();

        // TODO: have a "builder" for the model that automatically tracks the shape after each layer such that
        // we can just do model.prepare_input(&input).
        // Here is not possible since we didnt run through the onnx loader
        let input = Tensor::random(&input_shape);
        let input_padded = model.prepare_inputs(vec![input]).unwrap();
        let _ = model.run::<F>(&input_padded).unwrap();
    }

    #[test]
    fn test_model_manual_run() {
        let dense1 = Dense::<Element>::random(vec![
            10usize.next_power_of_two(),
            11usize.next_power_of_two(),
        ]);
        let dense2 = Dense::<Element>::random(vec![
            7usize.next_power_of_two(),
            dense1.ncols().next_power_of_two(),
        ]);
        let input_shape = vec![dense1.ncols()];
        let input = Tensor::<Element>::random(&input_shape);
        let output1 = evaluate_layer::<GoldilocksExt2, _, _>(&dense1, &vec![&input], None)
            .unwrap()
            .outputs()[0]
            .clone();
        let final_output = evaluate_layer::<GoldilocksExt2, _, _>(&dense2, &vec![&output1], None)
            .unwrap()
            .outputs()[0]
            .clone();

        let mut model = ProvableModel::<Element>::new_from_input_shapes(
            vec![input_shape],
            PaddingMode::NoPadding,
        );
        let first_id = model
            .add_consecutive_layer(Layer::Dense(dense1.clone()), None)
            .unwrap();
        let second_id = model
            .add_consecutive_layer(Layer::Dense(dense2.clone()), Some(first_id))
            .unwrap();
        model.route_output(None).unwrap();

        let trace = model.run::<F>(&vec![input]).unwrap();
        assert_eq!(trace.steps.len(), 2);
        // Verify first step
        assert_eq!(*trace.get_step(&first_id).unwrap().outputs()[0], output1);

        // Verify second step
        assert_eq!(
            *trace.get_step(&second_id).unwrap().outputs()[0],
            final_output.clone()
        );
        let (nrow, _) = (dense2.nrows(), dense2.ncols());
        assert_eq!(final_output.get_data().len(), nrow);
    }

    use ff::Field;
    #[test]
    fn test_model_sequential() {
        let (model, input) = ProvableModel::random(1).unwrap();
        model.describe();
        let trace = model.run::<F>(&input).unwrap().to_field();
        let dense_layers = model
            .to_unstable_iterator()
            .flat_map(|(id, l)| match l.operation {
                Layer::Dense(ref dense) => Some((*id, dense.clone())),
                _ => None,
            })
            .collect_vec();
        let matrices_mle = dense_layers
            .iter()
            .map(|(id, d)| (*id, d.matrix.to_mle_2d::<F>()))
            .collect_vec();
        assert_eq!(dense_layers.len(), 1);
        let point1 = random_bool_vector(dense_layers[0].1.matrix.nrows_2d().ilog2() as usize);
        let computed_eval1 = trace
            .get_step(&dense_layers[0].0)
            .expect(format!("Node with id {} not found", dense_layers[0].0).as_str())
            .outputs()[0]
            .get_data()
            .to_vec()
            .into_mle()
            .evaluate(&point1);
        let flatten_mat1 = matrices_mle[0].1.fix_high_variables(&point1);
        let bias_eval = dense_layers[0]
            .1
            .bias
            .evals_flat::<F>()
            .into_mle()
            .evaluate(&point1);
        let computed_eval1_no_bias = computed_eval1 - bias_eval;
        let input_vector = trace.input[0].clone();
        // since y = SUM M(j,i) x(i) + B(j)
        // then
        // y(r) - B(r) = SUM_i m(r,i) x(i)
        let full_poly = vec![
            flatten_mat1.clone().into(),
            input_vector.get_data().to_vec().into_mle().into(),
        ];
        let mut vp = VirtualPolynomial::new(flatten_mat1.num_vars());
        vp.add_mle_list(full_poly, F::ONE);
        #[allow(deprecated)]
        let (proof, _state) =
            IOPProverState::<F>::prove_parallel(vp.clone(), &mut default_transcript());
        let (p2, _s2) =
            IOPProverState::prove_batch_polys(1, vec![vp.clone()], &mut default_transcript());
        let given_eval1 = proof.extract_sum();
        assert_eq!(p2.extract_sum(), proof.extract_sum());
        assert_eq!(computed_eval1_no_bias, given_eval1);

        let _subclaim = IOPVerifierState::<F>::verify(
            computed_eval1_no_bias,
            &proof,
            &vp.aux_info,
            &mut default_transcript(),
        );
    }

    use crate::{Context, Prover, verify};
    use transcript::BasicTranscript;

    #[test]
    #[ignore = "This test should be deleted since there is no requant and it is not testing much"]
    fn test_single_matvec_prover() {
        let w1 = random_vector_quant(1024 * 1024);
        let conv1 = Tensor::new(vec![1024, 1024], w1.clone());
        let w2 = random_vector_quant(1024);
        let conv2 = Tensor::new(vec![1024], w2.clone());
        let input_shape = vec![1024];

        let mut model =
            ProvableModel::new_from_input_shapes(vec![input_shape], PaddingMode::Padding);
        let input = Tensor::random(&model.input_shapes()[0]);
        model
            .add_consecutive_layer(Layer::Dense(Dense::new(conv1, conv2)), None)
            .unwrap();
        model.route_output(None).unwrap();
        model.describe();
        let trace = model.run::<F>(&vec![input]).unwrap();
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let ctx =
            Context::<GoldilocksExt2>::generate(&model, None).expect("Unable to generate context");
        let io = trace.to_verifier_io();
        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_single_cnn_prover() {
        let n_w = 1 << 2;
        let k_w = 1 << 4;
        let n_x = 1 << 5;
        let k_x = 1 << 1;

        let in_dimensions: Vec<Vec<usize>> =
            vec![vec![k_x, n_x, n_x], vec![16, 29, 29], vec![4, 26, 26]];

        let conv1 = Tensor::random(&vec![k_w, k_x, n_w, n_w]);
        let input_shape = vec![k_x, n_x, n_x];

        let mut model =
            ProvableModel::new_from_input_shapes(vec![input_shape], PaddingMode::Padding);
        let input = Tensor::random(&model.input_shapes()[0]);
        let _conv_layer = model
            .add_consecutive_layer(
                Layer::Convolution(
                    Convolution::new(conv1.clone(), Tensor::random(&vec![conv1.kw()]))
                        .into_padded_and_ffted(&in_dimensions[0]),
                ),
                None,
            )
            .unwrap();
        model.route_output(None).unwrap();
        model.describe();
        let trace = model.run::<F>(&vec![input]).unwrap();
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let ctx =
            Context::<GoldilocksExt2>::generate(&model, None).expect("Unable to generate context");
        let io = trace.to_verifier_io();

        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");

        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_cnn_prover() {
        for i in 0..3 {
            for j in 2..5 {
                for l in 0..4 {
                    for n in 1..(j - 1) {
                        let n_w = 1 << n;
                        let k_w = 1 << l;
                        let n_x = 1 << j;
                        let k_x = 1 << i;

                        let in_dimensions: Vec<Vec<usize>> =
                            vec![vec![k_x, n_x, n_x], vec![16, 29, 29], vec![4, 26, 26]];
                        let input_shape = vec![k_x, n_x, n_x];
                        let conv1 = Tensor::random(&vec![k_w, k_x, n_w, n_w]);
                        let mut model = ProvableModel::<Element>::new_from_input_shapes(
                            vec![input_shape],
                            PaddingMode::Padding,
                        );
                        let input = Tensor::random(&model.input_shapes()[0]);
                        model
                            .add_consecutive_layer(
                                Layer::Convolution(
                                    Convolution::new(
                                        conv1.clone(),
                                        Tensor::random(&vec![conv1.kw()]),
                                    )
                                    .into_padded_and_ffted(&in_dimensions[0]),
                                ),
                                None,
                            )
                            .unwrap();
                        model.route_output(None).unwrap();
                        model.describe();
                        let trace = model.run::<F>(&vec![input]).unwrap();
                        let mut tr: BasicTranscript<GoldilocksExt2> =
                            BasicTranscript::new(b"m2vec");
                        let ctx = Context::<GoldilocksExt2>::generate(&model, None)
                            .expect("Unable to generate context");
                        let io = trace.to_verifier_io();
                        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>> =
                            Prover::new(&ctx, &mut tr);
                        let proof = prover.prove(trace).expect("unable to generate proof");
                        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
                            BasicTranscript::new(b"m2vec");
                        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
                    }
                }
            }
        }
    }
}
