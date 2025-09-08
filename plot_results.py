import matplotlib.pyplot as plt
import pandas as pd

# Load CSV
df = pd.read_csv("results.csv")

# Bar plot: mean Accuracy across models & tasks
mean_accuracy = df.groupby(["Model", "Task"])["Accuracy"].mean().unstack()*100

mean_accuracy.plot(
    kind="bar",
    figsize=(8, 6),
    title="Mean Accuracy across Models and Tasks"
)
plt.xlabel("Model")
plt.ylabel("Mean Accuracy [%]")
plt.legend(title="Task")
plt.ylim(0,100)
plt.grid()
plt.show()

# Per-model plots for Toolcalls and Belief
model_groups = df.groupby("Model")
for model_name, model_group in model_groups:
    print(f"Processing model: {model_name}")

    # Boxplot of Toolcalls per Task
    model_group.boxplot(column="Toolcalls", by="Task")
    plt.title(f"Toolcalls per Task for Model {model_name}")
    plt.suptitle("")
    plt.xlabel("Task")
    plt.ylabel("Toolcalls")
    plt.show()

    # Line plot comparing Belief across Tasks for this model
    pivot_belief = model_group.pivot(index="Trial", columns="Task", values="Belief")
    pivot_belief.plot(
        kind="line",
        marker="o",
        title=f"Belief across Tasks for Model {model_name}"
    )
    plt.xlabel("Trial")
    plt.ylabel("Belief")
    plt.grid()
    plt.show()
