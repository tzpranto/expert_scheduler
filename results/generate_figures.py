import json
from pathlib import Path
import matplotlib.pyplot as plt

# --- Update these if your filenames change ---
OLMOE_JSON = "./simulation_olmoe_1766153893.json"
GPT_JSON   = "./simulation_gpt5oss_1766153919.json"

def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def extract_series(data):
    """
    Returns:
      pools_pct: list[float]
      overall_lru: list[float]
      overall_lstm: list[float]
      stages: dict[str, dict[str, list[float]]]  # stages[policy][stage] = series
    """
    sims = data["simulations"]  
    pools_pct = []
    overall_lru = []
    overall_lstm = []

    # stage-wise
    stages = {"lru": {}, "lstm": {}}

    for k in sorted(sims.keys(), key=lambda x: int(x)):
        entry = sims[k]
        lru_res = entry["lru"]["results"]
        lstm_res = entry["lstm"]["results"]

        pools_pct.append(lru_res["pool_percentage"])
        overall_lru.append(lru_res["overall"]["hit_rate"])
        overall_lstm.append(lstm_res["overall"]["hit_rate"])

        for policy_name, res in [("lru", lru_res), ("lstm", lstm_res)]:
            by_stage = res.get("by_stage", {})
            for stage_name, stage_vals in by_stage.items():
                stages[policy_name].setdefault(stage_name, [])
                stages[policy_name][stage_name].append(stage_vals["hit_rate"])

    return pools_pct, overall_lru, overall_lstm, stages

def plot_overall(model_name, pools_pct, lru, lstm, outpath):
    plt.figure()
    plt.plot(pools_pct, lru, marker="o", label="LRU")
    plt.plot(pools_pct, lstm, marker="o", label="LSTM+LRU")
    plt.xlabel("Pool size (% of total experts)")
    plt.ylabel("Overall hit rate")
    plt.title(f"{model_name}: Overall hit rate vs pool size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=600)
    plt.close()

def plot_stages(model_name, pools_pct, stages, outpath):
    """
    stages: dict with keys "lru" and "lstm", each is stage->series
    Makes one plot with multiple lines: LRU(stage) and LSTM(stage).
    """
    plt.figure()

    # plot LRU stage lines
    for stage_name, series in stages["lru"].items():
        plt.plot(pools_pct, series, marker="o", label=f"LRU-{stage_name}")

    # plot LSTM stage lines
    for stage_name, series in stages["lstm"].items():
        plt.plot(pools_pct, series, marker="o", linestyle="--", label=f"LSTM-{stage_name}")

    plt.xlabel("Pool size (% of total experts)")
    plt.ylabel("Hit rate")
    plt.title(f"{model_name}: Stage-wise hit rate vs pool size")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=600)
    plt.close()

def main():
    out_dir = Path("./")
    out_dir.mkdir(parents=True, exist_ok=True)

    # OLMoE
    olmoe = load_results(OLMOE_JSON)
    pools, lru, lstm, stages = extract_series(olmoe)
    plot_overall("OLMoE", pools, lru, lstm, out_dir / "olmoe_overall_hit_rate.pdf")
    plot_stages("OLMoE", pools, stages, out_dir / "olmoe_stage_hit_rate.pdf")

    # GPT-OSS-20B
    gpt = load_results(GPT_JSON)
    pools, lru, lstm, stages = extract_series(gpt)
    plot_overall("GPT-OSS-20B", pools, lru, lstm, out_dir / "gpt_overall_hit_rate.pdf")
    plot_stages("GPT-OSS-20B", pools, stages, out_dir / "gpt_stage_hit_rate.pdf")

    print("Saved figures to:", out_dir)

if __name__ == "__main__":
    main()
