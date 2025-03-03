use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::BufReader,
    path::Path,
    time,
};

use anyhow::{Context as CC, ensure};
use clap::Parser;
use csv::WriterBuilder;
use goldilocks::GoldilocksExt2;
use itertools::Itertools;
use log::info;

use serde::{Deserialize, Serialize};
use zkml::{
    Context, Element, IO, Prover, argmax, default_transcript, load_mlp,
    lookup::LogUp,
    quantization::{Quantizer, TensorFielder},
    tensor::Tensor,
    verify,
};

use rmp_serde::encode::to_vec_named;

type F = GoldilocksExt2;

#[derive(Parser, Debug)]
struct Args {
    /// onxx file to load
    #[arg(short, long)]
    onnx: String,
    /// input / output vector file in JSON. Format "{ input_data: [a,b,c], output_data: [c,d] }"
    #[arg(short, long)]
    io: String,
    /// File where to write the benchmarks
    #[arg(short,long,default_value_t = {"bench.csv".to_string()})]
    bench: String,
    /// Maximum number of samples to process (default: all samples)
    #[arg(short, long)]
    max_samples: Option<usize>,
}
pub fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    run(args).context("error running bench:")?;
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct InputJSON {
    input_data: Vec<Vec<f64>>,
    output_data: Vec<Vec<f64>>,
}

impl InputJSON {
    /// Returns all (input,output) pairs from the path
    pub fn from(path: &str) -> anyhow::Result<Vec<(Vec<Element>, Vec<Element>)>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let u: Self = serde_json::from_reader(reader)?;
        u.validate()?;
        Ok(u.to_elements())
    }

    // Validate input data
    fn validate(&self) -> anyhow::Result<()> {
        let rrange = -1.0..=1.0;
        ensure!(!self.input_data.is_empty(), "Input data cannot be empty");
        ensure!(
            self.input_data.len() == self.output_data.len(),
            "Input and output data must have the same number of samples"
        );

        // Check if inputs contain values in valid range
        let input_isreal = self.input_data[0].iter().all(|v| rrange.contains(v));
        ensure!(
            input_isreal,
            "can only support real model so far (input at least)"
        );
        Ok(())
    }

    fn to_elements(self) -> Vec<(Vec<Element>, Vec<Element>)> {
        let mut result = Vec::with_capacity(self.input_data.len());

        for (input, output) in self
            .input_data
            .into_iter()
            .zip(self.output_data.into_iter())
        {
            let inputs = input
                .into_iter()
                .map(|e| Element::from_f32_unsafe(&(e as f32)))
                .collect_vec();

            let outputs = output
                .into_iter()
                .map(|e| Element::from_f32_unsafe(&(e as f32)))
                .collect_vec();

            result.push((inputs, outputs));
        }

        result
    }
}

const CSV_SAMPLE: &str = "sample";
const CSV_LOAD: &str = "load (ms)";
const CSV_SETUP: &str = "setup (ms)";
const CSV_INFERENCE: &str = "inference (ms)";
const CSV_PROVING: &str = "proving (ms)";
const CSV_VERIFYING: &str = "verifying (ms)";
const CSV_ACCURACY: &str = "accuracy (bool)";
const CSV_PROOF_SIZE: &str = "proof size (KB)";

fn run(args: Args) -> anyhow::Result<()> {
    // Load the model and context once for all samples
    info!("[+] Reading onnx model");
    let start_load = time::Instant::now();
    let model = load_mlp::<Element>(&args.onnx).context("loading model:")?;
    let load_time = start_load.elapsed().as_millis();
    model.describe();

    info!("[+] Reading input/output pairs from pytorch");
    let mut io_pairs = InputJSON::from(&args.io).context("loading input:")?;
    let total_samples = io_pairs.len();

    // Apply sample limit if specified
    if let Some(max_samples) = args.max_samples {
        if max_samples < total_samples {
            info!(
                "[+] Limiting to {} samples out of {} available",
                max_samples, total_samples
            );
            io_pairs.truncate(max_samples);
        }
    }

    let num_pairs = io_pairs.len();
    info!("[+] Processing {} input/output pairs", num_pairs);

    // Generate context for proving (done once)
    info!("[+] Generating context for proving");
    let start_setup = time::Instant::now();
    let ctx = Context::<F>::generate(&model).expect("unable to generate context");
    let setup_time = start_setup.elapsed().as_millis();

    let shape = model.input_shape();
    assert_eq!(shape.len(), 1, "only support vector as input for now");

    // Track total accuracy
    let mut correct_predictions = 0;
    let mut total_predictions = 0;

    // Process each input/output pair
    for (sample_idx, (input_vec, given_output)) in io_pairs.into_iter().enumerate() {
        info!("[+] Processing sample {}/{}", sample_idx + 1, num_pairs);

        let mut bencher = CSVBencher::from_headers(vec![
            CSV_SAMPLE,
            CSV_LOAD,
            CSV_SETUP,
            CSV_INFERENCE,
            CSV_PROVING,
            CSV_VERIFYING,
            CSV_PROOF_SIZE,
            CSV_ACCURACY,
        ]);

        // Set the sample index and reuse load/setup times
        bencher.set(CSV_SAMPLE, sample_idx);
        bencher.set(CSV_LOAD, load_time);
        bencher.set(CSV_SETUP, setup_time);

        // Prepare input tensor
        let input = Tensor::<Element>::new(vec![input_vec.len()], input_vec);
        let input = model.prepare_input(input);

        // Run inference
        info!("[+] Running inference for sample {}", sample_idx);
        let trace = bencher.r(CSV_INFERENCE, || model.run(input.clone()));
        let output = trace.final_output().clone();
        bencher.set(
            CSV_ACCURACY,
            compare(&given_output, &output.get_data().to_vec()),
        );

        // Run prover
        info!("[+] Running prover for sample {}", sample_idx);
        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _, LogUp>::new(&ctx, &mut prover_transcript);
        let proof = bencher.r(CSV_PROVING, move || {
            prover.prove(trace).expect("unable to generate proof")
        });

        // Calculate proof size
        let proof_bytes = to_vec_named(&proof)?;
        let proof_size_kb = proof_bytes.len() as f64 / 1024.0;
        bencher.set(CSV_PROOF_SIZE, format!("{:.3}", proof_size_kb));

        // Run verifier
        info!("[+] Running verifier for sample {}", sample_idx);
        let mut verifier_transcript = default_transcript();
        let io = IO::new(input.to_fields(), output.clone().to_fields());
        bencher.r(CSV_VERIFYING, || {
            verify::<_, _, LogUp>(ctx.clone(), proof, io, &mut verifier_transcript)
                .expect("invalid proof")
        });
        info!("[+] Verify proof: valid for sample {}", sample_idx);

        // Update accuracy tracking
        let is_correct = compare(&given_output, &output.get_data().to_vec());
        if is_correct == 1 {
            correct_predictions += 1;
        }
        total_predictions += 1;

        // Write results to CSV
        bencher.flush(&args.bench)?;
        info!(
            "[+] Benchmark results for sample {} appended to {}",
            sample_idx, args.bench
        );
    }

    // Print total accuracy at the end
    let accuracy_percentage = (correct_predictions as f64 / total_predictions as f64) * 100.0;
    info!(
        "[+] Overall accuracy: {}/{} correct predictions ({:.2}%)",
        correct_predictions, total_predictions, accuracy_percentage
    );

    Ok(())
}

fn compare<A: PartialOrd, B: PartialOrd>(given_output: &[A], computed_output: &[B]) -> usize {
    let a_max = argmax(given_output);
    let b_max = argmax(computed_output);
    info!("Accuracy: {}", if a_max == b_max { 1 } else { 0 });
    if a_max == b_max { 1 } else { 0 }
}

type Ms = u128;

struct CSVBencher {
    data: HashMap<String, String>,
    headers: Vec<String>,
}

impl CSVBencher {
    pub fn from_headers<S: IntoIterator<Item = T>, T: Into<String>>(headers: S) -> Self {
        let strings: Vec<String> = headers.into_iter().map(Into::into).collect();
        Self {
            data: Default::default(),
            headers: strings,
        }
    }

    pub fn r<A, F: FnOnce() -> A>(&mut self, column: &str, f: F) -> A {
        self.check(column);
        let now = time::Instant::now();
        let output = f();
        let elapsed = now.elapsed().as_millis();
        info!("STEP: {} took {}ms", column, elapsed);
        self.data.insert(column.to_string(), elapsed.to_string());
        output
    }

    fn check(&self, column: &str) {
        if self.data.contains_key(column) {
            panic!("CSVBencher only handles one row for now");
        }
        if !self.headers.contains(&column.to_string()) {
            panic!("column {} non existing", column);
        }
    }

    pub fn set<I: ToString>(&mut self, column: &str, data: I) {
        self.check(column);
        self.data.insert(column.to_string(), data.to_string());
    }

    fn flush(&self, fname: &str) -> anyhow::Result<()> {
        let file_exists = Path::new(fname).exists();
        let file = OpenOptions::new()
            .create(true)
            .append(file_exists)
            .write(true)
            .open(fname)?;
        let mut writer = WriterBuilder::new()
            .has_headers(!file_exists)
            .from_writer(file);

        let values: Vec<_> = self
            .headers
            .iter()
            .map(|k| self.data[k].to_string())
            .collect();

        if !file_exists {
            writer.write_record(&self.headers)?;
        }

        writer.write_record(&values)?;
        writer.flush()?;
        Ok(())
    }
}
