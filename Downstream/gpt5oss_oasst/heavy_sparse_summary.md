# Expert Routing Report (GPT5OSS OASST)

## Setup
- Traces dir: `gpt5oss/oasst`
- Files analyzed: **498** trace_*.json
- Layers: **24**, Experts: **32**
- Cutoff: top/bottom **10%** per selection frequency

## Key findings (overall tokens)
- Tokens: **54708**, top-1 routing events: **1312992**
- Heavy hitters (most used experts): 24 (7.04%), 23 (5.73%), 11 (4.31%), 10 (4.24%)
- Sparse hitters (least used experts): 25 (1.4181%), 20 (1.5133%), 29 (1.6133%), 3 (1.7941%)

### MoE perspective (interpretation)
- **Imbalance**: A few experts (24, 23, 11, 10) dominate routing (~21.3% combined), suggesting specialization or over-reliance; may merit load-balancing or routing regularization.
- **Under-utilization**: Experts 25, 20, 29, 3 sit at ~1.4–1.8%, indicating capacity headroom and potential to encourage diversity (e.g., auxiliary losses, temperature on gate, or dropout).
- **No analysis/final segments detected**: All tokens were classified as prompt; future traces with channel markers would enable per-stage routing insights.

## Visuals
![Heavy hitters](charts\heavy_hitters.png)
![Sparse hitters](charts\sparse_hitters.png)

## How to reproduce
```bash
# 1) Compute hitters
python analyze_hitters.py --traces-dir gpt5oss/oasst --output heavy_sparse_report.json
# 2) Render this report
python render_hitters_report.py
```