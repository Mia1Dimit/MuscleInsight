import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory to store plots if it doesn't exist
output_dir = 'output_plots'
os.makedirs(output_dir, exist_ok=True)

# Load the data
data = pd.read_csv("updated_data_with_phases.csv")

# Ensure columns are of the correct type (numeric), replacing non-numeric values with NaN
for col in ['rms_min', 'rms_max', 'iemg_min', 'iemg_max', 'mnf_min', 'mnf_max', 'mpf_min', 'mpf_max']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Remove rows where any of the metrics are NaN
data = data.dropna(subset=['rms_min', 'rms_max', 'iemg_min', 'iemg_max', 'mnf_min', 'mnf_max', 'mpf_min', 'mpf_max'])

# Define metrics for plotting
metrics = [
    ('rms_min', 'rms_max', 'RMS', 'RMS (Root Mean Square)'),
    ('iemg_min', 'iemg_max', 'IEMG', 'IEMG (Integrated EMG)'),
    ('mnf_min', 'mnf_max', 'MNF', 'MNF (Mean Frequency)'),
    ('mpf_min', 'mpf_max', 'MPF', 'MPF (Median Power Frequency)')
]

# Function to plot participant-specific line plots
def plot_participant_line_plots(metric_min, metric_max, title, ylabel):
    fig, axes = plt.subplots(len(data['Participant'].unique()), 1, figsize=(12, 16), sharex=True)
    for i, participant in enumerate(data['Participant'].unique()):
        participant_data = data[data['Participant'] == participant]
        axes[i].plot(participant_data['Session'], participant_data[metric_min], color='blue', label='Min')
        axes[i].plot(participant_data['Session'], participant_data[metric_max], color='red', label='Max')
        axes[i].set_title(f"{participant} - {title}")
        axes[i].set_ylabel(ylabel)
        axes[i].legend()
    fig.suptitle(f"Participant-Specific Line Plots - {title}", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'participant_line_plot_{title}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# Function to plot heatmaps by phase
def plot_heatmaps():
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 16))
    for i, (metric_min, metric_max, title, _) in enumerate(metrics):
        # Use groupby to aggregate data by Phase and Participant, and calculate the mean
        metric_data = data.groupby(['Phase', 'Participant'])[metric_min].mean().unstack()
        sns.heatmap(metric_data, annot=True, cmap="YlOrRd", ax=axes[i])
        axes[i].set_title(f"Heatmap - {title} by Phase")
        axes[i].set_ylabel('Participant')
    fig.suptitle("Heatmaps of Metrics Across Phases", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'heatmaps_by_phase.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# Function to plot small multiples for each phase
def plot_small_multiples():
    fig, axes = plt.subplots(len(metrics), len(data['Phase'].unique()), figsize=(16, 12), sharex=True, sharey=True)
    for i, (metric_min, metric_max, title, _) in enumerate(metrics):
        for j, phase in enumerate(data['Phase'].unique()):
            phase_data = data[data['Phase'] == phase]
            axes[i, j].scatter(phase_data['Session'], phase_data[metric_min], color='blue', label='Min')
            axes[i, j].scatter(phase_data['Session'], phase_data[metric_max], color='red', label='Max')
            axes[i, j].set_title(f"{title} - {phase}")
            axes[i, j].set_ylabel(f"{title}")
            if i == 0:
                axes[i, j].set_title(f"{phase}")
            if j == 0:
                axes[i, j].set_ylabel(f"{title}")
    fig.suptitle("Small Multiples - Metrics by Phase", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'small_multiples.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# Function to plot boxplot with participants
def plot_boxplot_with_participants(metric_min, title, ylabel, ax):
    ax.boxplot([data[data['Phase'] == 'Phase A (Rest)'][metric_min],
                data[data['Phase'] == 'Phase B (Activation)'][metric_min],
                data[data['Phase'] == 'Phase C (Active)'][metric_min]])
    ax.set_title(f'{title} - Box Plot')
    ax.set_xticklabels(['Phase A (Rest)', 'Phase B (Activation)', 'Phase C (Active)'])
    ax.set_ylabel(ylabel)

# Create all the figures
# Box plots
fig_box, axes_box = plt.subplots(4, 1, figsize=(10, 16))
for i, (metric_min, metric_max, title, ylabel) in enumerate(metrics):
    plot_boxplot_with_participants(metric_min, title, ylabel, axes_box[i])
fig_box.tight_layout()
fig_box.suptitle("Box Plots of Metrics Across Phases", fontsize=16)
fig_box.subplots_adjust(top=0.95)

# Save box plots
plt.savefig(os.path.join(output_dir, 'box_plots.png'), dpi=300, bbox_inches='tight')
plt.close(fig_box)

# Participant-specific line plots
for metric_min, metric_max, title, ylabel in metrics:
    plot_participant_line_plots(metric_min, metric_max, title, ylabel)

# Heatmaps
plot_heatmaps()

# Small multiples
plot_small_multiples()

print(f"All plots have been saved in the '{output_dir}' directory.")
