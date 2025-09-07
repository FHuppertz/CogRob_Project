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

    # Plot Belief vs Truth for each Task
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
        plt.show()

    # Plot mean Toolcalls per Task for this model
    mean_toolcalls = model_group.groupby("Task")["Toolcalls"].mean()
    mean_toolcalls.plot(
        kind="bar",
        title=f"Average Toolcalls per Task for Model {model_name}",
        legend=False
    )
    plt.xlabel("Task")
    plt.ylabel("Average number of Toolcalls")
    plt.show()
