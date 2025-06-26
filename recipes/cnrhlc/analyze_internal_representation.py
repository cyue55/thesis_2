# analyze_models_internal_representation.py

import argparse
import logging
import os
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from mbchl.has import HARegistry
from mbchl.training.ema import EMARegistry
from mbchl.training.losses import ControllableNoiseReductionHearingLossCompensationLoss
from mbchl.utils import read_yaml

# Reuse constant definitions from evaluate_debug.py
standard_audiograms = {
    "NH": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "N1": [10, 10, 10, 10, 10, 10, 15, 20, 30, 40],
    "N2": [20, 20, 20, 22.5, 25, 30, 35, 40, 45, 50],
    "N3": [35, 35, 35, 35, 40, 45, 50, 55, 60, 65],
    "N4": [55, 55, 55, 55, 55, 60, 65, 70, 75, 80],
    "N5": [65, 67.5, 70, 72.5, 75, 80, 80, 80, 80, 80],
    "N6": [75, 77.5, 80, 82.5, 85, 90, 90, 95, 100, 100],
    "N7": [90, 92.5, 95, 100, 105, 105, 105, 105, 105, 105],
    "S1": [10, 10, 10, 10, 10, 10, 15, 30, 55, 70],
    "S2": [20, 20, 20, 22.5, 25, 35, 55, 75, 95, 95],
    "S3": [30, 30, 35, 47.5, 60, 70, 75, 80, 80, 85],
}
audiogram_freqs = [250, 375, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000]

def peaknorm(x):
    return x / x.abs().max()

def setup_output_directories(base_dir):
    """Create output directory structure"""
    dirs = {
        'ir': os.path.join(base_dir, 'internal_representations'),  # Internal representation images
        'audio': os.path.join(base_dir, 'processed_audio'),        # Processed audio files
        'energy': os.path.join(base_dir, 'energy_distributions'),  # Energy distribution plots
        'comparison': os.path.join(base_dir, 'model_comparisons')  # Model comparison plots
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def get_model_shortname(model_name):
    """Generate concise model name"""
    # Special handling for "No Average" case
    if model_name == "No Average":
        return "no_avg"
        
    # Split string
    parts = model_name.split('-')
    
    # Process algorithm type
    algo_type = parts[0].lower()  # "rms" or "mean"
    
    # Get window size with ms unit (only the first number)
    window_size = parts[1]  # e.g., "32ms" from "32ms-16ms"
    if '_' in window_size:
        window_size = window_size.split('_')[0]
    if '-' in window_size:
        window_size = window_size.split('-')[0]
    if 'ms' not in window_size:
        window_size += 'ms'
    
    # Combine new name
    return f"{algo_type}_{window_size}"

def compare_model_representations(models_dict, signal, audiogram, profile, output_dir, fs, segment_len=1600):
    """Compare internal representations of different models"""
    # Set font sizes
    TITLE_SIZE = 24
    SUBTITLE_SIZE = 20
    LABEL_SIZE = 16
    TICK_SIZE = 14
    
    # 重新排序模型
    ordered_models = {
        "No Average": models_dict.get("No Average", None),
        "Mean-1ms": models_dict.get("Mean-1ms-0.5ms", None),
        "RMS-1ms": models_dict.get("RMS-1ms-0.5ms", None),
        "Mean-8ms": models_dict.get("Mean-8ms-4ms", None),
        "RMS-8ms": models_dict.get("RMS-8ms-4ms", None),
        "Mean-32ms": models_dict.get("Mean-32ms-16ms", None),
        "RMS-32ms": models_dict.get("RMS-32ms-16ms", None)
    }
    
    # 移除None值
    ordered_models = {k: v for k, v in ordered_models.items() if v is not None}
    
    fig, axes = plt.subplots(len(ordered_models), 3, figsize=(24, 5*len(ordered_models)))
    
    for i, (model_name, model_data) in enumerate(ordered_models.items()):
        model = model_data["model"]
        
        # Get internal representations
        noisy_nh = model._loss.am_nh(signal)
        noisy_hi = model._loss.am_hi(signal, audiogram=audiogram)
        
        # Process signal through model
        if model._audiogram:
            # Use audiogram from standard_audiograms
            thresholds = standard_audiograms[profile]
            current_audiogram = torch.tensor(
                list(zip(audiogram_freqs, thresholds)), device=signal.device
            )
            extra_inputs = [current_audiogram]
        else:
            extra_inputs = None
            
        output = model.enhance(signal.unsqueeze(0), extra_inputs=extra_inputs, use_amp=False)
        
        # Get compensated internal representation
        comp_hi = model._loss.am_hi(output[0], audiogram=audiogram)
        
        # Plot internal representations
        im1 = axes[i, 0].imshow(noisy_nh[:, :segment_len].cpu().numpy(), aspect="auto", origin="lower")
        im2 = axes[i, 1].imshow(noisy_hi[:, :segment_len].cpu().numpy(), aspect="auto", origin="lower")
        im3 = axes[i, 2].imshow(comp_hi[:, :segment_len].cpu().numpy(), aspect="auto", origin="lower")
        
        # Set titles with larger font and bold
        axes[i, 0].set_title(f"{model_name}: NH Internal Rep.", 
                            fontsize=SUBTITLE_SIZE, weight='bold')
        axes[i, 1].set_title(f"{model_name}: {profile} Internal Rep.", 
                            fontsize=SUBTITLE_SIZE, weight='bold')
        axes[i, 2].set_title(f"{model_name}: Compensated Internal Rep.", 
                            fontsize=SUBTITLE_SIZE, weight='bold')
        
        # Add axis labels
        for ax in axes[i]:
            ax.set_xlabel("Time (Samples)", fontsize=LABEL_SIZE, weight='bold')
            ax.set_ylabel("Frequency Channel", fontsize=LABEL_SIZE, weight='bold')
            ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    
    plt.tight_layout()
    comparison_filename = f"model_comparison_{profile.lower()}.png"
    plt.savefig(os.path.join(output_dir['comparison'], comparison_filename), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data for further analysis
    return {
        model_name: {
            "nh": model_data["model"]._loss.am_nh(signal).cpu().numpy(),
            "hi": model_data["model"]._loss.am_hi(signal, audiogram=audiogram).cpu().numpy() if profile != "NH" else model_data["model"]._loss.am_nh(signal).cpu().numpy(),
            "comp": model_data["model"]._loss.am_hi(
                model_data["model"].enhance(signal.unsqueeze(0), 
                    extra_inputs=[torch.tensor(list(zip(audiogram_freqs, standard_audiograms[profile])), device=signal.device)],
                    use_amp=False
                )[0], 
                audiogram=audiogram
            ).cpu().numpy() if profile != "NH" else model_data["model"]._loss.am_nh(
                model_data["model"].enhance(signal.unsqueeze(0), 
                    extra_inputs=[torch.tensor(list(zip(audiogram_freqs, standard_audiograms["NH"])), device=signal.device)],
                    use_amp=False
                )[0]
            ).cpu().numpy()
        } for model_name, model_data in models_dict.items()
    }

def visualize_single_model(model_name, model, signal, audiogram, profile, output_dir, fs, segment_len=1600):
    """Generate detailed visualization of internal representations for a single model"""
    # Set font sizes
    TITLE_SIZE = 24  # 主标题
    SUBTITLE_SIZE = 20  # 子标题
    LABEL_SIZE = 16  # 坐标轴标签
    TICK_SIZE = 14  # 刻度标签
    
    # Get internal representations
    noisy_nh = model._loss.am_nh(signal)
    noisy_hi = model._loss.am_hi(signal, audiogram=audiogram)
    
    # Process signal through model
    if model._audiogram:
        # Use audiogram from standard_audiograms
        thresholds = standard_audiograms[profile]
        current_audiogram = torch.tensor(
            list(zip(audiogram_freqs, thresholds)), device=signal.device
        )
        extra_inputs = [current_audiogram]
    else:
        extra_inputs = None
        
    output = model.enhance(signal.unsqueeze(0), extra_inputs=extra_inputs, use_amp=False)
    
    # Get compensated internal representation
    comp_hi = model._loss.am_hi(output[0], audiogram=audiogram)
    
    # Create image
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot internal representations
    im1 = axes[0].imshow(noisy_nh[:, :segment_len].cpu().numpy(), aspect="auto", origin="lower")
    im2 = axes[1].imshow(noisy_hi[:, :segment_len].cpu().numpy(), aspect="auto", origin="lower")
    im3 = axes[2].imshow(comp_hi[:, :segment_len].cpu().numpy(), aspect="auto", origin="lower")
    
    # Set titles with larger font and bold
    axes[0].set_title("Normal Hearing Internal Representation", 
                      fontsize=SUBTITLE_SIZE, weight='bold')
    axes[1].set_title(f"{profile} Hearing Loss Internal Representation", 
                      fontsize=SUBTITLE_SIZE, weight='bold')
    axes[2].set_title(f"Compensated {profile} Internal Representation", 
                      fontsize=SUBTITLE_SIZE, weight='bold')
    
    # Set global title
    fig.suptitle(f"Model: {model_name}, Profile: {profile}", 
                 fontsize=TITLE_SIZE, weight='bold')
    
    # Add axis labels
    for ax in axes:
        ax.set_xlabel("Time (Samples)", fontsize=LABEL_SIZE, weight='bold')
        ax.set_ylabel("Frequency Channel", fontsize=LABEL_SIZE, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    
    # Save image
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Use clearer file naming
    model_short = get_model_shortname(model_name)
    ir_filename = f"ir_{model_short}_{profile.lower()}.png"
    plt.savefig(os.path.join(output_dir['ir'], ir_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save processed audio
    audio_filename = f"enhanced_{model_short}_{profile.lower()}.wav"
    torchaudio.save(
        os.path.join(output_dir['audio'], audio_filename),
        peaknorm(output[0:1]).cpu(), # channel one hlc
        fs
    )
    
    return {
        "nh": noisy_nh.cpu().numpy(),
        "hi": noisy_hi.cpu().numpy(),
        "comp": comp_hi.cpu().numpy()
    }

def compare_models_energy_distribution(models_data, profile, output_dir):
    """Compare energy distributions of internal representations across models"""
    plt.figure(figsize=(24, 8))  # 增大图片尺寸
    
    # Set font sizes
    TITLE_SIZE = 20
    LABEL_SIZE = 16
    TICK_SIZE = 14
    LEGEND_SIZE = 14
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # 首先绘制其他模型（用虚线）
    for model_name, data in models_data.items():
        if model_name != "No Average":
            nh_energy = np.mean(data["nh"], axis=1)
            hi_energy = np.mean(data["hi"], axis=1)
            comp_energy = np.mean(data["comp"], axis=1)
            
            for ax_idx, energy in enumerate([nh_energy, hi_energy, comp_energy]):
                axes[ax_idx].plot(energy, 
                                label=model_name,
                                linewidth=2,
                                linestyle='--',
                                alpha=0.7)  # 略微降低其他线条的不透明度
    
    # 最后绘制No Average（用粗实线），确保它在最上层
    if "No Average" in models_data:
        baseline_data = models_data["No Average"]
        nh_energy = np.mean(baseline_data["nh"], axis=1)
        hi_energy = np.mean(baseline_data["hi"], axis=1)
        comp_energy = np.mean(baseline_data["comp"], axis=1)
        
        for ax_idx, energy in enumerate([nh_energy, hi_energy, comp_energy]):
            axes[ax_idx].plot(energy, 
                            label="No Average (Baseline)",
                            color='red',  # 使用醒目的红色
                            linewidth=4,  # 加粗线条
                            linestyle='-',
                            zorder=10)  # 确保在最上层
    
    # Set titles and labels with larger font
    axes[0].set_title("NH Internal Rep. Energy", fontsize=TITLE_SIZE, weight='bold')
    axes[1].set_title(f"{profile} Internal Rep. Energy", fontsize=TITLE_SIZE, weight='bold')
    axes[2].set_title(f"Compensated {profile} Internal Rep. Energy", fontsize=TITLE_SIZE, weight='bold')
    
    for ax in axes:
        ax.set_xlabel("Frequency Channel", fontsize=LABEL_SIZE, weight='bold')
        ax.set_ylabel("Average Energy", fontsize=LABEL_SIZE, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax.legend(fontsize=LEGEND_SIZE, loc='upper right')  # 调整图例位置
        ax.grid(True, linestyle=":", alpha=0.3)  # 降低网格线的存在感
    
    plt.tight_layout()
    energy_filename = f"energy_distribution_{profile.lower()}.png"
    plt.savefig(os.path.join(output_dir['energy'], energy_filename), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Debug mode: use custom arguments
    use_custom_args = True  # Set to False to use command line arguments
    
    if use_custom_args:
        # Set custom arguments for debugging
        custom_args = [
            "--models",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1/checkpoints/last.ckpt",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-integration-1-0.5/checkpoints/last.ckpt",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-integration-8-4/checkpoints/last.ckpt",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-integration-32-16/checkpoints/last.ckpt",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-integration-mean-1-0.5/checkpoints/last.ckpt",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-integration-mean-8-4/checkpoints/last.ckpt",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-integration-mean-32-16/checkpoints/last.ckpt",
            "--model_names",
            "No Average",
            "RMS-1ms-0.5ms",
            "RMS-8ms-4ms",
            "RMS-32ms-16ms",
            "Mean-1ms-0.5ms",
            "Mean-8ms-4ms",
            "Mean-32ms-16ms",
            "--testset",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/testset/",
            "--cfg_files",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1/config.yaml",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-integration-1-0.5/config.yaml",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-integration-8-4/config.yaml",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-integration-32-16/config.yaml",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-integration-mean-1-0.5/config.yaml",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-integration-mean-8-4/config.yaml",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-cnrhlc-l1-integration-mean-32-16/config.yaml",
            "--output_dir",
            "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/ir_analysis_results",
            "--profiles",
            "NH",
            "N1",
            "N2",
            "N3",
            "N4",
            "N5",
            "N6",
            "N7",
            "S1",
            "S2",
            "S3",
            "--segment_len",
            "1600"
        ]
        # Pass custom arguments to parser
        sys.argv = [sys.argv[0]] + custom_args
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Path to model checkpoints to analyze. Can specify multiple.")
    parser.add_argument("--model_names", nargs="+", 
                        help="Names for each model (for plot labels). If not provided, will use directory names.")
    parser.add_argument("--testset", required=True,
                        help="Path to test dataset")
    parser.add_argument("--cfg_files", nargs="+",
                        help="Path to config files for each model. If not provided, will try to auto-detect.")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--output_dir", default="ir_analysis", 
                        help="Output directory for analysis results")
    parser.add_argument("--profiles", nargs="+", default=["NH", "N3", "N7"], 
                        help="Audiogram profiles to analyze")
    parser.add_argument("--segment_len", type=int, default=1600,
                        help="Length of signal segment to visualize (in samples)")
    
    args = parser.parse_args()
    
    # Create output directory structure
    output_dirs = setup_output_directories(args.output_dir)
    
    # Process model names
    if args.model_names is None:
        args.model_names = [os.path.basename(os.path.dirname(os.path.dirname(m))) for m in args.models]
    
    if len(args.model_names) != len(args.models):
        raise ValueError("Number of model_names must match number of models")
    
    # Set device
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Build model dictionary
    models = {}
    for i, (model_path, model_name) in enumerate(zip(args.models, args.model_names)):
        logging.info(f"Loading model {i+1}/{len(args.models)}: {model_name}")
        
        # Load config
        if args.cfg_files and i < len(args.cfg_files):
            cfg_path = args.cfg_files[i]
        else:
            # Auto find config file
            dirname = os.path.dirname(model_path)
            cfg_path = os.path.join(dirname, "..", "config.yaml")
            if not os.path.exists(cfg_path):
                cfg_path = os.path.join(dirname, "config.yaml")
                if not os.path.exists(cfg_path):
                    raise FileNotFoundError(f"Config file not found for model {model_name}")
        
        logging.info(f"  Using config file: {cfg_path}")
        cfg = read_yaml(cfg_path)
        
        # Initialize model
        model = HARegistry.init(cfg["ha"], **cfg["ha_kw"])
        model.to(device)
        
        # Load checkpoint
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        
        # Load EMA
        if "ema" in cfg["trainer"] and cfg["trainer"]["ema"] is not None:
            ema_kw = cfg["trainer"]["ema_kw"]
            ema_cls = EMARegistry.get(cfg["trainer"]["ema"])
            ema_obj = ema_cls(model, **(ema_kw or {}))
            ema_obj.load_state_dict(state["ema"])
            ema_obj.apply()
        
        # Disable gradients and set to evaluation mode
        torch.set_grad_enabled(False)
        model.eval()
        
        # Check if model uses CNRHLC loss
        if not isinstance(model._loss, ControllableNoiseReductionHearingLossCompensationLoss):
            logging.warning(f"Model {model_name} does not use ControllableNoiseReductionHearingLossCompensationLoss")
            continue
        
        models[model_name] = {"model": model, "cfg": cfg}
    
    # Load test data
    logging.info(f"Loading test data from {args.testset}")
    in_files = [f for f in os.listdir(args.testset) if f.endswith("_mix.wav")]
    in_files = [f.split("_")[0] for f in in_files]
    in_files = [(f"{f}_mix.wav", f"{f}_target.wav") for f in sorted(in_files)]
    in_files = in_files[:1]  # Only take first file for analysis
    
    f = in_files[0]
    logging.info(f"Processing file: {f[0]}")
    x, fs = torchaudio.load(os.path.join(args.testset, f[0]))
    y, fs = torchaudio.load(os.path.join(args.testset, f[1]))
    x, y = x.to(device), y.to(device)
    
    # Save input signal
    torchaudio.save(os.path.join(args.output_dir, "input.wav"), peaknorm(x).cpu(), fs)
    torchaudio.save(os.path.join(args.output_dir, "target.wav"), peaknorm(y).cpu(), fs)
    
    # Process each hearing loss level
    all_results = {}
    for profile in args.profiles:
        logging.info(f"Processing profile: {profile}")
        thresholds = standard_audiograms[profile]
        audiogram = torch.tensor(
            list(zip(audiogram_freqs, thresholds)), device=device
        )
        
        # Compare internal representations of different models
        model_comparisons = compare_model_representations(
            models, x[0], audiogram, profile, output_dirs, fs, args.segment_len
        )
        
        # Save separate visualization for each model
        model_results = {}
        for model_name, model_data in models.items():
            logging.info(f"  Generating detailed visualization for model: {model_name}")
            model_results[model_name] = visualize_single_model(
                model_name, model_data["model"], x[0], audiogram, profile, 
                output_dirs, fs, args.segment_len
            )
        
        # Compare energy distributions
        compare_models_energy_distribution(model_results, profile, output_dirs)
        
        # Save results
        all_results[profile] = model_results
    
    # Optional: Save all results data for further analysis
    np.savez(os.path.join(args.output_dir, "analysis_results.npz"), results=all_results)
    
    logging.info("Analysis complete! Results saved to " + args.output_dir)