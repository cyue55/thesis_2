import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Set plot style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

MODEL_PATHS = {
    # "c3b": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1/scores.npz",
    "baseline": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1/scores_1.npz",
    "c5": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c5-tg2/scores_1.npz",
    "c3a": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c3a-tg2/scores_1.npz",
    # "c3b": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c3b-tg2/scores.npz",
    # "nr-only": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-nr-l1/scores.npz",
    # "hlc-only": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-hlc-l1/scores.npz",
    # "c2": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c2-tg2/scores.npz",
    # "c3a": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c3a-tg2/scores.npz",
    "c3b": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c3b-tg2/scores.npz",
    # "c4-8Hz": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c4-tg2-8hz/scores.npz",
    "c4": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c4-tg2-new/scores.npz",
    # "c4-20Hz": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c4-tg2-20hz/scores.npz",
    # "c4-32Hz": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c4-tg2-32hz/scores.npz",
    # "c4-48Hz": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c4-tg2-48hz/scores.npz",
    # "c4-72Hz": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c4-tg2-72hz/scores.npz",
    # "c4-12Hz":"/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c4-tg2-new/scores.npz",
    # "c5": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-phi-tg2/scores.npz",
    # "c6": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c6-tg2/scores.npz",
    # "RMS-1ms": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/lp/bsrnn-cnrhlc-l1-integration-1-0.5/scores.npz",
    # "RMS-8ms": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/lp/bsrnn-cnrhlc-l1-integration-8-4/scores.npz",

    # "Mean-1ms": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/lp/bsrnn-cnrhlc-l1-integration-mean-1-0.5/scores.npz",
    # "Mean-8ms": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/lp/bsrnn-cnrhlc-l1-integration-mean-8-4/scores.npz",
    "c8": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c8-12hz/scores.npz",
    "c9": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c9-12hz/scores.npz",
    # "c10": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-c10/scores.npz",
}
# Reorder models to put NR Only and No Average first
# MODEL_PATHS = {
#     "NR Only": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-nr-l1/scores.npz",
#     "HLC Only": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-hlc-l1/scores.npz",
#     "Joint": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1/scores.npz",
#     "RMS-1ms": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/lp/bsrnn-cnrhlc-l1-integration-1-0.5/scores.npz",
#     # "RMS-2ms": "models/bsrnn-cnrhlc-l1-integration-2-1-nalr/scores.npz",
#     # "RMS-4ms": "models/bsrnn-cnrhlc-l1-integration-4-2-nalr/scores.npz",
#     "RMS-8ms": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/lp/bsrnn-cnrhlc-l1-integration-8-4/scores.npz",
#     # "RMS-16ms": "models/bsrnn-cnrhlc-l1-integration-16-8-nalr/scores.npz",
#     "RMS-32ms": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/lp/bsrnn-cnrhlc-l1-integration-32-16/scores.npz",
#     "Mean-1ms": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/lp/bsrnn-cnrhlc-l1-integration-mean-1-0.5/scores.npz",
#     "Mean-8ms": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/lp/bsrnn-cnrhlc-l1-integration-mean-8-4/scores.npz",
#     "Mean-32ms": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/lp/bsrnn-cnrhlc-l1-integration-mean-32-16/scores.npz",
#     "Dual Constant Time": "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-dtc-l1/scores.npz",
# }
# Define profiles and metrics
PROFILES = ["NH", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "S1", "S2", "S3"]
METRICS = ["haspi", "hasqi", "pesq", "estoi", "snr"]

# Define colors for each model
COLORS = {
    'baseline': '#000000',   # 黑色
    # 'c3b_new': '#008000',   
    'nr-only': '#808080',   # 灰色
    'hlc-only': '#A9A9A9',   # 深灰色
    'c2': '#FF4500',     # 橙红色
    'c3a': '#E41A1C',    # 红色
    'c3b': '#377EB8',    # 蓝色
    'c4-12Hz': '#008000',     # 绿色
    'c4-8Hz': '#32CD32',     # 浅绿色
    'c4-20Hz': '#006400',     # 深绿色
    'c4-32Hz': '#0000FF',     # 蓝色
    'c4-48Hz': '#0000FF',     # 蓝色
    'c4-72Hz': '#0000FF',     # 蓝色
    'c4': '#4DAF4A',     # 绿色
    'c5': '#0000FF',     # 蓝色
    'c6': '#800080',     # 紫色
    'c8': '#984EA3',     # 紫色
    'c9': '#FF7F00',     # 橙色
    'c10': '#0000FF',     # 蓝色
    # 'RMS-1ms': '#800080',    # 紫色
    # 'RMS-8ms': '#FF1493',    # 深粉红色
    # 'Mean-1ms': '#4B0082',   # 靛蓝色
    # 'Mean-8ms': '#FF8C00'    # 深橙色
}

# Define markers for each model
MARKERS = {
    'baseline': '*',
    # 'c3b_new': 'o',
    'nr-only': 'o',
    'hlc-only': 's',
    'c2': 'o',
    'c3a': 's',
    'c3b': 'D',
    'c4-32Hz': 'D',
    'c4-48Hz': 'o',
    'c4-72Hz': 's',
    'c4-8Hz': 'v',      # 倒三角
    'c4-12Hz': '^',     # 正三角
    'c4-20Hz': '<',     # 左三角
    'c4': 'v',
    'c5': 'D',
    'c6': 'o',
    'c8': 'D',
    'c9': 'o',
    'c10': 's',
    # 'RMS-1ms': 'o',
    # 'RMS-8ms': 's',
    # 'Mean-1ms': '^',
    # 'Mean-8ms': 'D'
}

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
            if os.path.exists(full_path):
                scores = np.load(full_path)
                print(f"\nDebug info for {model_name}:")
                print(f"Available keys in scores: {scores.files}")
                if 'haspi.NH' in scores:
                    print(f"haspi.NH shape: {scores['haspi.NH'].shape}")
                if 'hasqi.NH' in scores:
                    print(f"hasqi.NH shape: {scores['hasqi.NH'].shape}")
                scores_dict[model_name] = scores
                print(f"Successfully loaded scores for {model_name}")
            else:
                print(f"Scores file not found for {model_name}: {full_path}")
        except Exception as e:
            print(f"Failed to load scores for {model_name}: {e}")
    return scores_dict

def plot_metric(scores_dict, metric, alpha_idx, alpha_value, output_dir):
    """Plot comparison for each metric for a specific alpha value"""
    plt.figure(figsize=(15, 8))
    
    has_data = False  # 用于跟踪是否有数据被绘制
    
    # Plot line for each model
    for model_name, scores in scores_dict.items():
        means = []
        print(f"\nProcessing {model_name} for {metric}:")
        for profile in PROFILES:
            key = f"{metric}.{profile}"
            if key in scores:
                try:
                    data = scores[key]
                    print(f"  {key} shape: {data.shape}")
                    all_samples_mean = data[:, alpha_idx].mean()
                    means.append(all_samples_mean)
                    print(f"  Mean value: {all_samples_mean}")
                except (IndexError, KeyError) as e:
                    print(f"  Error processing {key}: {e}")
                    continue
            else:
                print(f"  Key {key} not found in scores")
        
        # 绘制数据
        if means:  # 确保有数据才绘制
            has_data = True
            plt.plot(range(len(PROFILES)), means, 
                    label=f"{model_name} (α={alpha_value:.1f})",
                    color=COLORS[model_name],
                    marker=MARKERS[model_name],
                    markersize=10,
                    linewidth=3,
                    linestyle='-',
                    alpha=0.8)
            print(f"  Plotted {len(means)} points for {model_name}")
        else:
            print(f"  No data points to plot for {model_name}")

    if not has_data:
        print(f"No data available for {metric} at alpha={alpha_value:.1f}")
        plt.close()
        return

    # Customize plot style with full metric name and alpha value
    plt.title(f'{METRIC_FULL_NAMES[metric]}\nScores Across Hearing Profiles (α={alpha_value:.1f})', fontsize=20, pad=20, weight='bold')
    plt.xlabel('Hearing Profile', fontsize=16)
    plt.ylabel(f'{metric.upper()} Score', fontsize=16)
    
    # Set x-axis ticks
    plt.xticks(range(len(PROFILES)), PROFILES, rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    
    # Adjust legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=16)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set base path
    base_path = "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc"
    
    # Alpha values from 0 to 1
    alpha_values = np.linspace(0, 1, 11)
    
    # Load all scores
    print("Loading model scores...")
    scores_dict = load_scores(base_path)
    
    if not scores_dict:
        print("No scores were loaded. Please check if the score files exist.")
        return
    
    # Generate plots for each alpha value
    for alpha_idx, alpha_value in enumerate(alpha_values):
        # Create directory for this alpha value
        alpha_dir = os.path.join(base_path, f"metrics_analysis_results_new/alpha_{alpha_value:.1f}")
        os.makedirs(alpha_dir, exist_ok=True)
        
        print(f"\nProcessing alpha = {alpha_value:.1f}")
        # Generate plots for each metric
        for metric in METRICS:
            print(f"Generating comparison plot for {metric}...")
            plot_metric(scores_dict, metric, alpha_idx, alpha_value, alpha_dir)
    
    print(f"Analysis complete! Results saved in metrics_analysis_results/")

if __name__ == "__main__":
    main()
