import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Set plot style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Reorder models to put NR Only and No Average first
MODEL_PATHS = {
    "NR Only": "models/bsrnn-nr-l1-nalr/scores.npz",
    "No Average(cnrhlc, α=1)": "models/bsrnn-cnrhlc-l1-nalr/scores.npz",
    "RMS-1ms": "models/bsrnn-cnrhlc-l1-integration-1-0.5-nalr/scores.npz",
    # "RMS-2ms": "models/bsrnn-cnrhlc-l1-integration-2-1-nalr/scores.npz",
    # "RMS-4ms": "models/bsrnn-cnrhlc-l1-integration-4-2-nalr/scores.npz",
    "RMS-8ms": "models/bsrnn-cnrhlc-l1-integration-8-4-nalr/scores.npz",
    # "RMS-16ms": "models/bsrnn-cnrhlc-l1-integration-16-8-nalr/scores.npz",
    "RMS-32ms": "models/bsrnn-cnrhlc-l1-integration-32-16-nalr/scores.npz",
    "Mean-1ms": "models/bsrnn-cnrhlc-l1-integration-mean-1-0.5-nalr/scores.npz",
    "Mean-8ms": "models/bsrnn-cnrhlc-l1-integration-mean-8-4-nalr/scores.npz",
    "Mean-32ms": "models/bsrnn-cnrhlc-l1-integration-mean-32-16-nalr/scores.npz",
}

# Define profiles and metrics
PROFILES = ["NH", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "S1", "S2", "S3"]
METRICS = ["haspi", "hasqi", "pesq", "estoi", "snr"]

# Define colors with distinct colors for NR Only and No Average
COLORS = ['#0000FF',  # Blue for NR Only
          '#000000',  # Black for No Average
          '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f']

# Define markers with distinct markers for NR Only and No Average
MARKERS = ['*',       # Star for NR Only
          'X',       # X for No Average
          'o', 's', '^', 'v', 'D', 'p', 'h', '8']

# 添加指标全称的映射
METRIC_FULL_NAMES = {
    "haspi": "Hearing-Aid Speech Perception Index (HASPI)",
    "hasqi": "Hearing-Aid Speech Quality Index (HASQI)",
    "pesq": "Perceptual Evaluation of Speech Quality (PESQ)",
    "estoi": "Extended Short-Time Objective Intelligibility (ESTOI)",
    "snr": "Signal-to-Noise Ratio (SNR)"
}

def load_scores(base_path):
    """Load scores for all models"""
    scores_dict = {}
    for model_name, rel_path in MODEL_PATHS.items():
        full_path = os.path.join(base_path, rel_path)
        try:
            scores = np.load(full_path)
            scores_dict[model_name] = scores
            print(f"Successfully loaded scores for {model_name}")
        except Exception as e:
            print(f"Failed to load scores for {model_name}: {e}")
    return scores_dict

def plot_metric(scores_dict, metric, output_dir):
    """Plot comparison for each metric"""
    plt.figure(figsize=(15, 8))
    
    # Plot line for each model
    for (model_name, scores), color, marker in zip(scores_dict.items(), COLORS, MARKERS):
        means = []
        for profile in PROFILES:
            key = f"{metric}.{profile}"
            if key in scores:
                means.append(scores[key][:, 0].mean())
        
        # Set different line styles and marker sizes for NR Only and No Average
        if model_name in ["NR Only", "No Average(cnrhlc, α=1)"]:
            markersize = 14
            linewidth = 4
            linestyle = '-'  # 改为实线
            label = model_name
        else:
            markersize = 10
            linewidth = 3
            linestyle = '--'  # 其他模型改为虚线
            label = f"{model_name} (α=1)"
        
        # Plot lines and markers
        plt.plot(range(len(PROFILES)), means, 
                label=label,
                color=color,
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.8)

    # Customize plot style with full metric name
    plt.title(f'{METRIC_FULL_NAMES[metric]}\nScores Across Hearing Profiles (NAL-R)', 
             fontsize=20, 
             pad=20,
             weight='bold')
    plt.xlabel('Hearing Profile', fontsize=16)
    plt.ylabel(f'{metric.upper()} Score', fontsize=16)
    
    # Set x-axis ticks
    plt.xticks(range(len(PROFILES)), PROFILES, rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    
    # Adjust legend
    plt.legend(bbox_to_anchor=(1.05, 1), 
              loc='upper left', 
              borderaxespad=0.,
              fontsize=16)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric}_comparison_nalr.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def main():
    # Set base path
    base_path = "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc"
    output_dir = os.path.join(base_path, "nalr_analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all scores
    print("Loading model scores...")
    scores_dict = load_scores(base_path)
    
    # Generate plots for each metric
    for metric in METRICS:
        print(f"Generating comparison plot for {metric}...")
        plot_metric(scores_dict, metric, output_dir)
    
    print(f"Analysis complete! Results saved in {output_dir}")

if __name__ == "__main__":
    main()
