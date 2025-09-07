import matplotlib.pyplot as plt
import pandas as pd

task_results = pd.read_csv("results.csv").groupby("Task")

# Plot the beleif vs truth for each Task and every experiment in the tasks
for i in range(len(task_results.groups)):
    task_group = task_results.get_group(i+1)
    task_group[["Experiment","belief", "truth"]].plot(x="Experiment", kind="line", linestyle="dotted", marker="o", title=f"Belief values for Task {i+1}")


# Plotting the mean of all toolcalls for each task
mean_toolcalls = task_results.mean(numeric_only=True)
mean_toolcalls.plot(y="Toolcalls", kind="bar", title="Mean Toolcalls of Evaluation")
plt.legend(["Toolcalls"])


plt.show()
