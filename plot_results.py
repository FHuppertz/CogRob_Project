import matplotlib.pyplot as plt
import pandas as pd
import os

# Make sure "plots" directory exists
os.makedirs("plots", exist_ok=True)

# Load CSV
df = pd.read_csv("results.csv")

# ============================
# Loop over memory states
# ============================
memory_groups = df.groupby("Memory")
for memory_state, mem_group in memory_groups:
    print(f"Processing Memory state: {memory_state}")

    # ============================
    # Bar plot: mean Accuracy per Model and Task
    # ============================
    mean_accuracy = mem_group.groupby(["Model", "Task"])["Accuracy"].mean().unstack() * 100.0

    ax = mean_accuracy.plot(
        kind="bar",
        figsize=(8, 6),
        title=f"Mean Accuracy across Models and Tasks (Memory {memory_state})"
    )
    plt.xlabel("Model")
    plt.ylabel("Mean Accuracy [%]")
    plt.legend(title="Task")

    # Y-axis ticks
    yticks = [i for i in range(0, 101, 10)]
    plt.yticks(yticks)
    plt.ylim(0.0, 100.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3)

    plt.savefig(f"plots/mean_accuracy_memory_{memory_state}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ============================
    # Per-model Boxplots for Toolcalls
    # ============================
    model_groups = mem_group.groupby("Model")
    for model_name, model_group in model_groups:
        model_group.boxplot(column="Toolcalls", by="Task")
        plt.title(f"Toolcalls per Task for Model {model_name} (Memory {memory_state})")
        plt.suptitle("")
        plt.xlabel("Task")
        plt.ylabel("Toolcalls")

        plt.savefig(f"plots/toolcalls_memory_{memory_state}_model_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close()
