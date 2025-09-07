import matplotlib.pyplot as plt
import pandas as pd

# Load CSV and group by Model
df = pd.read_csv("results.csv")
model_groups = df.groupby("Model")

# Iterate over each model
for model_name, model_group in model_groups:
    print(f"Processing model: {model_name}")

    # Group by Task within this model
    task_groups = model_group.groupby("Task")

    # Plot Belief vs Truth (line) for each Task
    for task_number, task_group in task_groups:
        task_group.plot(
            x="Experiment",
            y=["Belief", "Truth"],
            kind="line",
            linestyle="dotted",
            marker="o",
            title=f"Belief vs Truth for Model {model_name}, Task {task_number}"
        )
        plt.xlabel("Experiment")
        plt.ylabel("Values")


    # Create a boxplot of Toolcalls per Task
    model_group.boxplot(column="Toolcalls", by="Task")

    plt.title(f"Toolcalls per Task for Model {model_name}")
    plt.suptitle("")  # remove the automatic 'Boxplot grouped by Task' title
    plt.xlabel("Task")
    plt.ylabel("Toolcalls")

    # Line plot comparing Belief across Tasks for this model
    pivot_belief = model_group.pivot(index="Experiment", columns="Task", values="Belief")
    pivot_belief.plot(
        kind="line",
        marker="o",
        title=f"Belief across Tasks for Model {model_name}"
    )
    plt.xlabel("Experiment")
    plt.ylabel("Belief")

    # Show on screen
    plt.show()
