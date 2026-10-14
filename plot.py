import os
import pandas as pd
import matplotlib.pyplot as plt

# (file suffix, CPU label, GPU label, line style)
MACHINES = [
    ("1080", "Ryzen 5 3600", "GTX 1080", "-"),
    ("t4",   "AWS Xeon",     "Tesla T4", "--"),
]

fig, ax = plt.subplots(figsize=(9, 5.5))

for suffix, cpu_name, gpu_name, style in MACHINES:
    cpu_file = f"cpu_results_{suffix}.csv"
    gpu_file = f"gpu_results_{suffix}.csv"

    if os.path.exists(cpu_file):
        cpu = pd.read_csv(cpu_file).groupby("N")["time_ms"].median()
        ax.plot(cpu.index, cpu.values, style, marker="o", color="tab:blue",
                label=f"CPU ({cpu_name})")

    if os.path.exists(gpu_file):
        gpu = pd.read_csv(gpu_file).groupby("N").median()
        ax.plot(gpu.index, gpu["kernel_ms"], style, marker="s", color="tab:orange",
                label=f"Naive CUDA kernel ({gpu_name})")
        ax.plot(gpu.index, gpu["total_ms"], style, marker="^", color="tab:green",
                label=f"Naive CUDA kernel + copies ({gpu_name})")

ax.set_xlabel("Matrix size N")
ax.set_ylabel("Time (ms, log scale)")
ax.set_yscale("log")
ax.set_title("Matrix multiply runtime vs. N")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("runtime_vs_N.png", dpi=150)
print("wrote runtime_vs_N.png")
