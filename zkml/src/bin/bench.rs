use std::{fs::File, io::BufReader, path::Path};

use anyhow::{ensure, Context as CC};
use clap::Parser;
use goldilocks::GoldilocksExt2;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use zkml::{default_transcript, load_mlp, lookup::{LogUp,LookupProtocol}, vector_to_field_par, verify, Context, Element, Prover, IO};

type F = GoldilocksExt2;

#[derive(Parser,Debug)]
struct Args {
    /// onxx file to load
    #[arg(short, long)]
    onnx: String,    
    /// input vector file in JSON. Format "{ input: [a,b,c] }"
    #[arg(short,long)]
    input: String,
    /// File where to write the benchmarks
    #[arg(short,long,default_value_t = {"bench.csv".to_string()})]
    bench: String,
}
pub fn main() -> anyhow::Result<()> {
    print!("Hello world");
    let args = Args::parse();
    run(args).context("error running bench:")?;
    Ok(())
}

#[derive(Serialize,Deserialize)]
struct InputJSON {
    input: Vec<f64>,
}

impl InputJSON {
    pub fn from(path: &str) -> anyhow::Result<Vec<Element>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let u :Self = serde_json::from_reader(reader)?;
        u.validate()?;
        u.to_elements()
    }
    // poor's man validation
    fn validate(&self) -> anyhow::Result<()> {
        let rrange = (-1.0,1.0);
        let isreal = self.input.iter().all(|v| *v >= rrange.0 && *v <= rrange.1);
        ensure!(isreal ,"can only support either quant or real");
        Ok(())
    }
    fn to_elements(self) -> anyhow::Result<Vec<Element>> {
        Ok(self.input.into_iter().map(|e| e as Element).collect_vec())
    }
}

fn run(args: Args) -> anyhow::Result<()> {
    let model = load_mlp::<Element>(&args.onnx)?;
    let input = InputJSON::from(&args.input)?;
    let ctx = Context::<F>::generate(&model).expect("unable to generate context");
    let shape = model.input_shape();
    assert_eq!(shape.len(), 1,"only support vector as input for now");

    let trace = model.run(input.clone());
    let output = trace.final_output().to_vec();
    println!("[+] Run inference. Result: {:?}", output);

    let mut prover_transcript = default_transcript();
    let prover = Prover::<_, _, LogUp<F>>::new(&ctx, &mut prover_transcript);
    println!("[+] Run prover");
    let proof = prover.prove(trace).expect("unable to generate proof");

    let mut verifier_transcript = default_transcript();
    let io = IO::new(vector_to_field_par(&input), output.to_vec());
    verify::<_, _, LogUp<F>>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
    println!("[+] Verify proof: valid");

    Ok(())
}