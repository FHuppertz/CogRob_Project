import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results.csv").groupby("Task")

task_group = df.get_group(2)

print(task_group)

task_group[["Experiment","belief", "truth"]].plot(x="Experiment", kind="line", marker="o", title="Belief values for Task 2")


mean_toolcalls = df.mean(numeric_only=True)
mean_toolcalls.plot(y="Toolcalls", kind="bar", title="Mean Toolcalls of Evaluation")
plt.legend(["Toolcalls"])
plt.show()
