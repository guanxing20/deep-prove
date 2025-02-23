use std::{collections::HashMap, fs::{File, OpenOptions}, io::BufReader, time};

use anyhow::{ensure, Context as CC};
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use goldilocks::GoldilocksExt2;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_json::Number;
use zkml::{argmax, default_transcript, load_mlp, lookup::LogUp, vector_to_field_par, verify, Context, Element, Prover, IO};
use zkml::quantization::Quantizer;

type F = GoldilocksExt2;

#[derive(Parser,Debug)]
struct Args {
    /// onxx file to load
    #[arg(short, long)]
    onnx: String,    
    /// input / output vector file in JSON. Format "{ input_data: [a,b,c], output_data: [c,d] }"
    #[arg(short,long)]
    io: String,
    /// File where to write the benchmarks
    #[arg(short,long,default_value_t = {"bench.csv".to_string()})]
    bench: String,
}
pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run(args).context("error running bench:")?;
    Ok(())
}

#[derive(Serialize,Deserialize)]
struct InputJSON {
    input_data: Vec<Vec<f64>>,
    output_data: Vec<Vec<f64>>,
}

impl InputJSON {
    /// Returns (input,output) from the path
    pub fn from(path: &str) -> anyhow::Result<(Vec<Element>,Vec<Element>)> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let u :Self = serde_json::from_reader(reader)?;
        u.validate()?;
        Ok(u.to_elements())
    }
    // poor's man validation
    fn validate(&self) -> anyhow::Result<()> {
        let rrange = (-1.0..=1.0);
        ensure!(self.input_data.len() == 1);
        let input_isreal = self.input_data[0].iter().all(|v| rrange.contains(v));
        let output_isreal = self.output_data[0].iter().all(|v| rrange.contains(v));
        ensure!(input_isreal && output_isreal ,"can only support real model so far (input + output)");
        Ok(())
    }
    fn to_elements(mut self) -> (Vec<Element>,Vec<Element>) {
        let inputs = self.input_data.remove(0).into_iter().map(|e| Element::from_f32_unsafe(&(e as f32))).collect_vec();
        let outputs = self.output_data.remove(0).into_iter().map(|e| Element::from_f32_unsafe(&(e as f32))).collect_vec();
        (inputs,outputs)
    }
}

fn run(args: Args) -> anyhow::Result<()> {
    let mut bencher = CSVBencher::from_headers(vec!["load_model","setup","inference","proving","verifying","accuracy"]);
    println!("[+] Reading onnx model");
    let model = bencher.r("load_model", || load_mlp::<Element>(&args.onnx)).context("loading model:")?;
    println!("[+] Reading input/output from pytorch");
    let (input,given_output) = InputJSON::from(&args.io).context("loading input:")?;

    println!("[+] Generating context for proving");
    let ctx = bencher.r("setup",|| Context::<F>::generate(&model).expect("unable to generate context"));
    let shape = model.input_shape();
    assert_eq!(shape.len(), 1,"only support vector as input for now");

    println!("[+] Running inference");
    let trace = bencher.r("inference",|| model.run(input.clone()));
    let output = trace.final_output().to_vec();
    bencher.set("accuracy", compare(&given_output,&output));
    
    println!("[+] Running prover");
    let mut prover_transcript = default_transcript();
    let prover = Prover::<_, _, LogUp<F>>::new(&ctx, &mut prover_transcript);
    let proof = bencher.r("proving",move || prover.prove(trace).expect("unable to generate proof"));

    println!("[+] Running verifier");
    let mut verifier_transcript = default_transcript();
    let io = IO::new(vector_to_field_par(&input), output.to_vec());
    bencher.r("verifying", || verify::<_, _, LogUp<F>>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof"));
    println!("[+] Verify proof: valid");

    bencher.flush(&args.bench)?;
    println!("[+] Benchmark results appended to {}",args.bench);
    Ok(())
}

fn compare<A: PartialOrd, B: PartialOrd>(given_output: &[A],computed_output: &[B]) -> usize {
    let a_max = argmax(given_output);
    let b_max = argmax(computed_output);
    if a_max == b_max { 1 } else { 0 }
}

type Ms = u128;

struct CSVBencher {
    data: HashMap<String,String>,
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

    pub fn r<A,F: FnOnce() -> A>(&mut self, column: &str, f: F) -> A{
        self.check(column);
        let now = time::Instant::now();
        let output = f();
        let elapsed = now.elapsed().as_millis();

        self.data.insert(column.to_string(), elapsed.to_string());
        output
    }

    fn check(&self, column: &str) {
        if self.data.contains_key(column) {
            panic!("CSVBencher only handles one row for now");
        }
        if !self.headers.contains(&column.to_string()) {
            panic!("column {} non existing",column);
        }
    }


    pub fn set<I: ToString>(&mut self, column: &str, data: I) {
        self.check(column);
        self.data.insert(column.to_string(), data.to_string());
    }

    fn flush(self,fname: &str) -> anyhow::Result<()> {
        let file = OpenOptions::new().create(true).append(true).open(&fname)?;
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);

        let keys: Vec<_> = self.data.keys().cloned().collect();
        let values: Vec<_> = keys.iter().map(|k| self.data[k].to_string()).collect();

        if ReaderBuilder::new().has_headers(true).from_path(&fname).is_err() {
            writer.write_record(&keys)?;
        }

        writer.write_record(&values)?;
        writer.flush()?;
        Ok(())
    }
}