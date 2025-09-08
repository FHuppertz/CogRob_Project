import matplotlib.pyplot as plt
import pandas as pd
import os

# Make sure "plots" directory exists
os.makedirs("plots", exist_ok=True)

# Load CSV
df = pd.read_csv("results.csv")

# ============================
# Bar plot: mean Accuracy
# ============================
mean_accuracy = df.groupby(["Model", "Task"])["Accuracy"].mean().unstack() * 100.0

ax = mean_accuracy.plot(
    kind="bar",
    figsize=(8, 6),
    title="Mean Accuracy across Models and Tasks"
)
plt.xlabel("Model")
plt.ylabel("Mean Accuracy [%]")
plt.legend(title="Task")

# Set custom ticks explicitly
yticks = [i for i in range(0, 101, 10)]  # ticks every 10%
plt.yticks(yticks)

plt.ylim(0.0, 100.0)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Add labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", padding=3)  # one decimal, small gap

# Save as PNG inside "plots" folder
plt.savefig("plots/mean_accuracy.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================
# Per-model plots
# ============================
model_groups = df.groupby("Model")
for model_name, model_group in model_groups:
    print(f"Processing model: {model_name}")

    # Boxplot of Toolcalls per Task
    model_group.boxplot(column="Toolcalls", by="Task")
    plt.title(f"Toolcalls per Task for Model {model_name}")
    plt.suptitle("")
    plt.xlabel("Task")
    plt.ylabel("Toolcalls")

    # Save per-model boxplot as PNG inside "plots"
    plt.savefig(f"plots/toolcalls_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
