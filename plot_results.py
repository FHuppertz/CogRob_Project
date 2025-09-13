# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import numpy as np

# ============================
# 1. DATA LOADING AND PREPARATION
# ============================
# Create plots directory
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Load CSV file
try:
    df = pd.read_csv("results.csv")
except FileNotFoundError:
    print("Error: 'results.csv' not found. Please make sure the file is in the same directory.")
    exit()

# Map long model names to shorter names
model_mapping = {
    "qwen3-coder-480b-a35b-instruct": "Qwen3 Coder 480B",
    "gpt-4.1-2025-04-14": "GPT-4.1",
    "claude-sonnet-4-20250514": "Claude 4 Sonnet",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "deepseek-v3.1": "DeepSeek V3.1"
}
df['Model'] = df['Model'].map(model_mapping)

# ============================
# 2. PLOTTING FUNCTIONS
# ============================
def create_bar_plot(data, value_col, title, ytick_step, filename, groupby_cols=["Model", "Task"]):
    """
    Generates and saves a bar plot for a given value column.
    """
    ax = (data.groupby(groupby_cols)[value_col].mean() * 100).unstack().plot(
        kind="bar",
        figsize=(10, 6),
        title=title
    )
    plt.xlabel(groupby_cols[0])
    plt.ylabel(f"Mean {value_col} [%]")
    plt.legend(title=groupby_cols[1] if len(groupby_cols) > 1 else "Category")

    yticks = [i for i in range(0, 101, ytick_step)]
    plt.yticks(yticks)
    plt.ylim(0.0, 110.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300)
    plt.close()

def create_grouped_boxplot(model_name, model_group):
    """
    Generates and saves a grouped boxplot for a specific model.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Toolcalls per Memory State and Task for Model {model_name}")
    ax.set_xlabel("Memory State")
    ax.set_ylabel("Toolcalls")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    ytick_step_boxplot = 5
    # Check if model_group is empty to prevent errors
    if not model_group.empty:
      ymax = model_group['Toolcalls'].max()
      yticks = [i for i in range(0, int(ymax) + 1, ytick_step_boxplot)]
      ax.set_yticks(yticks)
    else:
      ax.set_yticks([])

    memory_states = sorted(model_group['Memory'].unique())
    tasks = sorted(model_group['Task'].unique())
    num_tasks = len(tasks)

    plot_data, x_positions, xtick_labels = [], [], []
    group_width = num_tasks + 1

    for i, memory in enumerate(memory_states):
        xtick_labels.append(memory)
        for j, task in enumerate(tasks):
            data = model_group[(model_group['Memory'] == memory) & (model_group['Task'] == task)]['Toolcalls']
            plot_data.append(data)
            x_positions.append(i * group_width + j)

    # Only create boxplot if there is data to plot
    if plot_data and any(len(d) > 0 for d in plot_data):
        bplots = ax.boxplot(plot_data, positions=x_positions, widths=0.8, patch_artist=True)
        colors = plt.cm.tab10.colors
        for i in range(len(plot_data)):
            bplots['boxes'][i].set_facecolor(colors[i % num_tasks])

        for median in bplots['medians']:
            median.set_color('black')

        for i, median in enumerate(bplots['medians']):
            x_pos = x_positions[i]
            y_pos = median.get_ydata()[0]
            ax.text(x_pos, y_pos, f"{y_pos:.0f}",
                    ha='center', va='center', fontsize=10, color='white',
                    fontweight='bold', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.2'))

        ax.legend(handles=[bplots['boxes'][i] for i in range(num_tasks)], labels=tasks, title="Task")

    xtick_positions = [i * group_width + (num_tasks - 1) / 2 for i in range(len(memory_states))]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"toolcalls_model_{model_name}_by_memory_and_task.png"), dpi=300)
    plt.close()

def create_violin_plot(data_df, x_col, value_col, title, filename):
    """
    Generates and saves a violin plot to show the distribution of a variable.
    """
    plt.figure(figsize=(12, 7))
    # Check if there are at least two unique hue levels to use `split`
    if data_df['Task'].nunique() >= 2:
      sns.violinplot(
          x=x_col,
          y=value_col,
          hue="Task",
          data=data_df,
          split=True,
          inner="quartile",
          palette="pastel"
      )
    else:
      sns.violinplot(
          x=x_col,
          y=value_col,
          hue="Task",
          data=data_df,
          inner="quartile",
          palette="pastel"
      )
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(f"Distribution of {value_col}")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300)
    plt.close()

def create_confusion_heatmap(data_df, model_name, filename):
    """
    Generates a confusion matrix heatmap for a given model, comparing Belief to Truth.
    Excludes rows where 'Stopped' == 1.
    """
    # Filter out rows where the run was stopped
    filtered_df = data_df[data_df['Stopped'] != 1].copy()

    if filtered_df.empty:
        print(f"Warning: No data for {model_name} with 'Stopped' != 1. Skipping confusion matrix plot.")
        return

    # Create a base 2x2 confusion matrix with zeros
    base_matrix = pd.DataFrame(
        np.zeros((2, 2)),
        index=pd.Index([0, 1], name='Actual Outcome (Truth)'),
        columns=pd.Index([0, 1], name='Predicted Outcome (Belief)')
    )

    # Calculate the raw confusion matrix from the filtered data
    raw_matrix = pd.crosstab(
        filtered_df['Truth'],
        filtered_df['Belief'],
        rownames=['Actual Outcome (Truth)'],
        colnames=['Predicted Outcome (Belief)']
    )

    # Add the raw counts to the base matrix
    for (truth, belief), count in raw_matrix.stack().items():
        base_matrix.loc[truth, belief] = count

    # Normalize the confusion matrix to get percentages
    row_sums = base_matrix.sum(axis=1)
    # Avoid division by zero for rows with no data by replacing sum with 1
    # where the sum is 0
    row_sums[row_sums == 0] = 1
    normalized_matrix = base_matrix.div(row_sums, axis=0) * 100

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        normalized_matrix,
        annot=True,
        fmt='.2f',  # Format to 2 decimal places for percentages
        cmap='Blues',
        xticklabels=['Failure', 'Success'],
        yticklabels=['Failure', 'Success']
    )
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Outcome (Belief) in %")
    plt.ylabel("Actual Outcome (Truth) in %")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300)
    plt.close()

def create_stopped_bar_plot(data_df, filename):
    """
    Generates a bar plot showing the percentage of stopped runs for each model, normalized to 100.
    """
    run_counts = data_df.groupby('Model').size().reset_index(name='Total Runs')
    stopped_counts = data_df[data_df['Stopped'] == 1].groupby('Model').size().reset_index(name='Stopped Runs')

    merged_counts = pd.merge(run_counts, stopped_counts, on='Model', how='left').fillna(0)
    merged_counts['Non-Stopped Runs'] = merged_counts['Total Runs'] - merged_counts['Stopped Runs']

    # Calculate percentages
    merged_counts['Stopped %'] = (merged_counts['Stopped Runs'] / merged_counts['Total Runs']) * 100
    merged_counts['Non-Stopped %'] = (merged_counts['Non-Stopped Runs'] / merged_counts['Total Runs']) * 100

    merged_counts_p = merged_counts.set_index('Model')

    ax = merged_counts_p[['Non-Stopped %', 'Stopped %']].plot(
        kind='bar',
        stacked=True,
        figsize=(10, 6),
        color=['#4CAF50', '#FF5722'],
        title='Proportion of Stopped Runs per Model'
    )

    plt.xlabel("Model")
    plt.ylabel("Proportion of Runs [%]")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Run Status", labels=['Success', 'Stopped'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 110)

    # Add percentage labels
    for container in ax.containers:
        labels = [f"{h:.1f}%" if h > 0 else "" for h in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold')


    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300)
    plt.close()


# ============================
# 3. MAIN EXECUTION
# ============================
def generate_all_plots(dataframe):
    print("Starting data analysis and plot generation.")

    # Create a filtered DataFrame excluding "Stopped" column
    # (The confusion matrix filters internally)
    filtered_df = dataframe[dataframe['Stopped'] != 1].copy()

    # Generate the bar plot for stopped runs
    print("Generating stopped runs bar plot.")
    create_stopped_bar_plot(dataframe, "stopped_runs_proportion.png")

    # Generate bar plots from the data - grouped by model
    for model_name, model_group in dataframe.groupby("Model"):
        print(f"Processing truth bar plots for Model: {model_name}")
        create_bar_plot(model_group, "Truth", f"Mean Ground Truth across Tasks and Memory States for {model_name}", 10, f"mean_truth_model_{model_name.replace(' ', '_')}.png", groupby_cols=["Task", "Memory"])

    # for model_name, model_group in filtered_df.groupby("Model"):
    #     print(f"Processing accuracy bar plots for Model: {model_name}")
    #     create_bar_plot(model_group, "Accuracy", f"Mean Accuracy across Tasks and Memory States for {model_name}", 10, f"mean_accuracy_model_{model_name.replace(' ', '_')}.png", groupby_cols=["Task", "Memory"])

    # Generate grouped boxplots and confusion heatmaps for each model
    for model_name, model_group in filtered_df.groupby("Model"):
        print(f"Generating plots for Model: {model_name}")
        # Pass the filtered data for the boxplot
        create_grouped_boxplot(model_name, model_group[model_group['Stopped'] != 1])
        # The confusion heatmap filters the data internally
        create_confusion_heatmap(model_group, model_name, f"confusion_matrix_{model_name.replace(' ', '_')}.png")

    # Generate violin plot for each model
    print("\nGenerating violin plots for each model.")
    for model_name, model_group in filtered_df.groupby("Model"):
        create_violin_plot(
            model_group,
            "Memory",
            "Toolcalls",
            f"Toolcalls Distribution by Memory State and Task for {model_name}",
            f"toolcalls_violin_plot_{model_name}.png"
        )
    print("Plot generation complete. Check the 'plots' directory for your images.")


    create_bar_plot(dataframe, "Truth", "Mean Truth across Models and Tasks", 10, "mean_truth_total.png")

# Run the plot generation
generate_all_plots(df)
