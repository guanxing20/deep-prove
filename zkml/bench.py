import subprocess
import os
import time
import sys
from pathlib import Path
import argparse
import shutil
import sys
import csv
import logging
import subprocess
from typing import Callable, Dict, List, Union
import re
import json

logging.basicConfig(level=logging.INFO)

class CSVBencher:
    def __init__(self, headers: List[str]):
        self.headers = headers
        self.data: Dict[str, str] = {}
    
    def _check(self, column: str):
        if column in self.data:
            raise ValueError("CSVBencher only handles one row for now")
        if column not in self.headers:
            raise ValueError(f"Column {column} does not exist")
    
    def r(self, column: str, task: Union[Callable, List[str]]):
        """Runs a function or a shell command, measures execution time, and stores the result."""
        self._check(column)
        start_time = time.perf_counter()
        
        if isinstance(task, list):
            result = subprocess.run(task, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Command {task} failed with exit code {result.returncode}: {result}")
        else:
            result = task()
        
        elapsed = int((time.perf_counter() - start_time) * 1000)  # Convert to ms
        logging.info(f"STEP: {column} took {elapsed}ms")
        self.data[column] = str(elapsed)
        return result
    
    def set(self, column: str, value: str):
        """Sets a value for a column after validation."""
        self._check(column)
        self.data[column] = str(value)
    
    def flush(self, fname: str):
        """Writes the collected data to a CSV file."""
        file_path = Path(fname)
        file_exists = file_path.exists()
        
        with file_path.open("a", newline="") as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(self.headers)  # Write headers if file is new
            
            values = [self.data.get(header, "") for header in self.headers]
            writer.writerow(values)
        
        logging.info(f"Flushed data to {fname}")


def ensure_command_exists(command):
    if shutil.which(command) is None:
        print(f"❌ Error: '{command}' is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)  # Exit with an error code

def ex(cmd):
    print(f"\n🔄 Running command: {' '.join(cmd)}")
    start_time = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.perf_counter()
    
    elapsed_time_ms = (end_time - start_time) * 1000
    
    if result.stdout:
        print(f"📝 Output:\n{result.stdout}")
    
    if result.returncode != 0:
        print(f"❌ Error (code {result.returncode}):\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    
    print(f"⏱️  Took {elapsed_time_ms:.0f}ms")
    
    return {
        "stdout": result.stdout,
        "elapsed_time_ms": elapsed_time_ms
    }

PYTORCH_SCRIPT = "./assets/scripts/MLP/mlp.py"
BENCH_FOLDER = Path("./bench/")
BENCH_ZKML = BENCH_FOLDER / "zkml.csv"
BENCH_EZKL = "ezkl.csv"
EZKL_KZG_PARAMS = "kzg.params"
MODEL = "mlp-model.onnx"
INPUT = "mlp-input.json"


def create_model(args):
    ex(["python3", PYTORCH_SCRIPT,
        "--num-dense", str(args.num_dense),
        "--layer-width", str(args.layer_width),
        "--export", str(BENCH_FOLDER)])

def run_zkml():
    ensure_command_exists("cargo")
    out = ex(["cargo", "run", "--release", "--", 
              "-i", str(BENCH_FOLDER / INPUT),
              "-o", str(BENCH_FOLDER / MODEL),
              "--bench", str(BENCH_ZKML)])
    print("out: ", out)

def run_ezkl():
    print(f"Moving to {BENCH_FOLDER} for ezkl bench")
    os.chdir(BENCH_FOLDER)
    ensure_command_exists("ezkl")
    SETUP = "setup (ms)" 
    INFERENCE = "inference (ms)"
    PROVING = "proving (ms)"
    VERIFYING = "verifying (ms)"
    ACCURACY = "accuracy (bool)"  # New column for correctness check
    
    bencher = CSVBencher([SETUP, INFERENCE, PROVING, VERIFYING, ACCURACY])
    
    # Run setup steps
    ex(["ezkl", "gen-settings", "-M", MODEL])
    ex(["ezkl", "calibrate-settings", "-M", MODEL, "-D", INPUT])
    if not Path(EZKL_KZG_PARAMS).exists():
        print("Downloading SRS params")
        ex(["ezkl", "get-srs", "--srs-path", EZKL_KZG_PARAMS])
    ex(["ezkl", "compile-circuit", "-M", MODEL])
    
    # Run setup and measure time
    bencher.r(SETUP, ["ezkl", "setup", "--srs-path", EZKL_KZG_PARAMS])
    
    # Run inference and measure time
    bencher.r(INFERENCE, ["ezkl", "gen-witness", "-D", INPUT])
    
    # For proving, extract the specific timing
    proving_result = ex(["ezkl", "prove", "--srs-path", EZKL_KZG_PARAMS])
    
    # Extract the proof time using regex
    proof_time_match = re.search(r"\[.*ezkl::pfsys\] - proof took (\d+\.\d+)", proving_result["stdout"])
    if proof_time_match:
        proof_time_seconds = float(proof_time_match.group(1))
        proof_time_ms = int(proof_time_seconds * 1000)
        print(f"Extracted proof time: {proof_time_ms}ms")
        bencher.set(PROVING, str(proof_time_ms))
    else:
        print("Could not extract proof time, using full command time")
        # Use the elapsed time from ex() function
        bencher.set(PROVING, str(int(proving_result["elapsed_time_ms"])))
    
    # Run verification and measure time
    bencher.r(VERIFYING, ["ezkl", "verify", "--srs-path", EZKL_KZG_PARAMS])
    
    # Extract rescaled_outputs from proof.json and compute argmax
    with open("proof.json", "r") as f:
        proof_data = json.load(f)
    
    ezkl_outputs = [float(x) for x in proof_data["pretty_public_inputs"]["rescaled_outputs"][0]]
    ezkl_argmax = ezkl_outputs.index(max(ezkl_outputs))
    print(f"EZKL output: {ezkl_outputs}, argmax: {ezkl_argmax}")
    
    # Extract PyTorch output from input.json and compute argmax
    with open(INPUT, "r") as f:
        input_data = json.load(f)
    
    pytorch_outputs = [float(x) for x in input_data["output_data"][0]]
    pytorch_argmax = pytorch_outputs.index(max(pytorch_outputs))
    print(f"PyTorch output: {pytorch_outputs}, argmax: {pytorch_argmax}")
    
    # Compare argmax values and set correctness
    is_correct = 1 if ezkl_argmax == pytorch_argmax else 0
    bencher.set(ACCURACY, str(is_correct))
    print(f"Correctness check: {'PASS' if is_correct else 'FAIL'}")
    
    bencher.flush(BENCH_EZKL)

def main():
    parser = argparse.ArgumentParser(description="mlp generator --num-dense and --layer-width")
    parser.add_argument("--num-dense", type=int, required=True, help="Number of dense layers")
    parser.add_argument("--layer-width", type=int, required=True, help="Width of each layer")

    args = parser.parse_args()
    print(f"num_dense: {args.num_dense}, layer_width: {args.layer_width}")

    folder = Path(BENCH_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)  # Creates folder if missing

    create_model(args)
    run_zkml()
    run_ezkl()


if __name__ == "__main__":
    main()