import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def perform_residual_analysis(json_path):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    N_actual = []
    T_decode = []
    TPOT_obs = []

    for run in data["raw_runs"]:
        if run["actual_tokens"] > 1 and run["decode_ms"] > 0:
            N_actual.append(run["actual_tokens"])
            T_decode.append(run["decode_ms"])
            TPOT_obs.append(run["decode_ms"] / (run["actual_tokens"] - 1))

    if not N_actual:
        print("No valid data points found.")
        return

    N = np.array(N_actual)
    T = np.array(T_decode)
    obs = np.array(TPOT_obs)

    # Analysis A: Linear Fit for T_decode = B1 * N + B0
    N_minus_1 = N - 1
    z = np.polyfit(N_minus_1, T, 1)
    beta_1 = z[0]  # Marginal cost per token (ms/token)
    beta_0 = z[1]  # Fixed overhead (ms)

    # R^2
    p = np.poly1d(z)
    T_pred = p(N_minus_1)
    ss_res = np.sum((T - T_pred) ** 2)
    ss_tot = np.sum((T - np.mean(T)) ** 2)
    r2_decode = 1 - (ss_res / ss_tot)

    # Analysis B: Calculate TPOT_predicted
    TPOT_pred = beta_1 + (beta_0 / N_minus_1)

    # Analysis C & D: Residuals (ei = obs - pred)
    residuals = obs - TPOT_pred

    print("\n=== Model Parameters ===")
    print(f"Beta_1 (Marginal cost) : {beta_1:.4f} ms/token")
    print(f"Beta_0 (Fixed overhead): {beta_0:.4f} ms")
    print(f"R^2 of T_decode        : {r2_decode:.6f}")

    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("TPOT Residual Analysis (Decoding Fixed Overhead vs. Physical Scaling)", fontsize=14)

    # 1. T_decode vs (N-1)
    ax1.scatter(N_minus_1, T, alpha=0.5, color='blue', label='Observed $T_{decode}$')
    x_range = np.linspace(N_minus_1.min(), N_minus_1.max(), 100)
    ax1.plot(x_range, p(x_range), "r--", label=f"Fit: $T = {beta_1:.2f}N' + {beta_0:.2f}$")
    ax1.set_title("Analysis A: Decode Time Fit")
    ax1.set_xlabel("Output Tokens - 1")
    ax1.set_ylabel("Decode Time (ms)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. TPOT vs (N-1)
    ax2.scatter(N_minus_1, obs, alpha=0.5, color='green', label='Observed TPOT')
    ax2.plot(x_range, beta_1 + (beta_0 / x_range), "r--", label=f"Predicted: $\\beta_1 + \\beta_0/N'$")
    ax2.axhline(beta_1, color='purple', linestyle=':', label=f"Asymptote ($\\beta_1$ = {beta_1:.2f})")
    ax2.set_title("Analysis B & C: TPOT Observed vs Predicted")
    ax2.set_xlabel("Output Tokens - 1")
    ax2.set_ylabel("TPOT (ms/token)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Residuals
    ax3.scatter(N_minus_1, residuals, alpha=0.5, color='orange')
    ax3.axhline(0, color='red', linestyle='--')
    ax3.set_title("Analysis D: Residuals ($e_i = TPOT_{obs} - TPOT_{pred}$)")
    ax3.set_xlabel("Output Tokens - 1")
    ax3.set_ylabel("Residual Error (ms/token)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/tpot_residuals.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform TPOT Residual Analysis")
    parser.add_argument("json_file", help="Path to the JSON results file")
    args = parser.parse_args()

    perform_residual_analysis(args.json_file)