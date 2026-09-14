import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def calculate_statistics(x, y):
    # 1. Slope & Intercept
    z = np.polyfit(x, y, 1)
    slope, intercept = z[0], z[1]

    # 2. Pearson r
    r_matrix = np.corrcoef(x, y)
    pearson_r = r_matrix[0, 1]

    # 3. R^2
    r_squared = pearson_r ** 2

    # 4. Mean & Std
    mean_y = np.mean(y)
    std_y = np.std(y)

    return slope, intercept, pearson_r, r_squared, mean_y, std_y

def plot_experiment_data(json_path):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    actual_tokens = []
    decode_times_sec = []
    tpots = []

    for run in data["raw_runs"]:
        if run["actual_tokens"] > 1 and run["decode_ms"] > 0 and run["tpot_ms"] is not None:
            actual_tokens.append(run["actual_tokens"])
            decode_times_sec.append(run["decode_ms"] / 1000.0)
            tpots.append(run["tpot_ms"])

    if not actual_tokens:
        print("No valid data points found to plot.")
        return

    x_arr = np.array(actual_tokens)
    y_decode = np.array(decode_times_sec)
    y_tpot = np.array(tpots)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"LLM Output Length Characterization (N={len(actual_tokens)})\nModel: {data['metadata']['model']}", fontsize=14)

    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)

    # Chart 1: Decode Time vs Output Length
    slope1, intcpt1, r1, r2_1, mean1, std1 = calculate_statistics(x_arr, y_decode)

    ax1.scatter(x_arr, y_decode, alpha=0.4, color='blue', edgecolors='k', label='Measured Runs')
    x_range1 = np.linspace(x_arr.min(), x_arr.max(), 100)
    ax1.plot(x_range1, slope1 * x_range1 + intcpt1, "r--", alpha=0.8, label='Linear Fit')

    stats_text1 = (
        f"Pearson $r$ : {r1:.4f}\n"
        f"$R^2$ : {r2_1:.4f}\n"
        f"Slope : {slope1:.4f} s/tok\n"
        f"Mean : {mean1:.3f} $\pm$ {std1:.3f} s"
    )
    ax1.text(0.05, 0.95, stats_text1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=bbox_props)

    ax1.set_title("Decode Time vs. Output Length")
    ax1.set_xlabel("Actual Output Tokens")
    ax1.set_ylabel("Decode Time (Seconds)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right')

    # Chart 2: TPOT vs Output Length
    slope2, intcpt2, r2, r2_2, mean2, std2 = calculate_statistics(x_arr, y_tpot)

    ax2.scatter(x_arr, y_tpot, alpha=0.3, color='green', edgecolors='k', label='Measured Runs')
    x_range2 = np.linspace(x_arr.min(), x_arr.max(), 100)
    ax2.plot(x_range2, slope2 * x_range2 + intcpt2, "r--", alpha=0.8, label='Linear Fit')

    stats_text2 = (
        f"Pearson $r$ : {r2:.4f}\n"
        f"$R^2$ : {r2_2:.4f}\n"
        f"Slope : {slope2:.5f} ms/tok\n"
        f"Mean : {mean2:.2f} $\pm$ {std2:.2f} ms"
    )
    ax2.text(0.05, 0.95, stats_text2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=bbox_props)

    ax2.set_title("TPOT vs. Output Length")
    ax2.set_xlabel("Actual Output Tokens")
    ax2.set_ylabel("Time Per Output Token (ms/token)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("figure/output.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Output Length Profiling Results with Statistics")
    parser.add_argument("json_file", help="Path to the JSON results file")
    args = parser.parse_args()

    plot_experiment_data(args.json_file)