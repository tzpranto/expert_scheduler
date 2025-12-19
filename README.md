# Expert Scheduler

Analyzing Feasibility and Effectiveness of Experts' Weight Scheduling for MoE LLMs in Memory-Constrained GPUs

## Setup

Create and activate conda environment:
```bash
conda create -n cse530 python=3.13.9
conda activate cse530
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Collect Traces

Generate execution traces for the models:

```bash
python trace_generator.py --model_id olmoe --max_token 256 --mode exec
python trace_generator.py --model_id gpt5oss --max_token 512 --mode exec
```
### 2. Trace Analysis

Generate analysis on heavy vs sparse hitters, entropy, heatmaps and prefill-vs-gen comparison. 
```bash
python analysis/gpt5oss_oasst/analyze_hitters.py
python analysis/gpt5oss_oasst/entropy_calc.py
python analysis/gpt5oss_oasst/render_hitters_report.py
python analysis/gpt5oss_oasst/plot_heatmaps.py
python analysis/compare_prefill_vs_gen.py
```




### 3. Train LSTM Model

Train prediction models on collected traces:

```bash
python prediction/train.py --model olmoe --data-dir ./moe_traces/olmoe/oasst/
python prediction/train.py --model gpt5oss --data-dir ./moe_traces/gpt5oss/oasst/
```

### 4. Run Simulator

Evaluate different expert pool configurations:

```bash
python prediction/run_simulation.py --model olmoe --pool-percentages 20 30 40 50 60 70 80
python prediction/run_simulation.py --model gpt5oss --pool-percentages 20 30 40 50 60 70 80
```

## Results

Simulation results will be saved in the ```results``` directory with performance metrics for each pool percentage configuration.