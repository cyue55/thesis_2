import torch
import numpy as np
import matplotlib.pyplot as plt
from mbchl.signal.auditory import FrameBasedAveraging

def visualize_frame_average():
    # 模拟训练环境的数据
    batch_size = 32
    channels = 31
    time = 64000
    fs = 16000  # 采样频率
    
    # 创建模拟数据（使用随机数据模拟真实信号）
    x = torch.randn(batch_size, channels, time)
    
    # 创建FrameBasedAveraging实例
    frame_avg = FrameBasedAveraging(
        fs=fs,
        window_size_ms=8,
        stride_ms=8,
        mode="rms"
    )
    
    # 获取处理后的结果
    y = frame_avg(x)
    
    # 选择第一个batch和第一个channel进行可视化
    batch_idx = 0
    channel_idx = 0
    
    # 创建时间轴
    t_input = np.arange(time) / fs
    t_output = np.arange(y.shape[-1]) * frame_avg.stride / fs
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 绘制输入信号
    plt.subplot(2, 1, 1)
    plt.plot(t_input[:int(1*fs)], x[batch_idx, channel_idx, :int(1*fs)].numpy())
    plt.title(f'Input Signal (Batch {batch_idx}, Channel {channel_idx})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # 绘制输出信号
    plt.subplot(2, 1, 2)
    plt.plot(t_output[:int(1*fs/frame_avg.stride)], 
             y[batch_idx, channel_idx, :int(1*fs/frame_avg.stride)].numpy())
    plt.title(f'Frame Average Output (RMS Mode)')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Value')
    plt.grid(True)
    
    # 添加统计信息
    stats_text = (
        f"Input shape: {x.shape}\n"
        f"Output shape: {y.shape}\n"
        f"Window size: {frame_avg.frame_size} samples ({frame_avg.frame_size/fs*1000:.1f} ms)\n"
        f"Stride: {frame_avg.stride} samples ({frame_avg.stride/fs*1000:.1f} ms)\n"
        f"Input mean: {x[batch_idx, channel_idx].mean():.4f}\n"
        f"Output mean: {y[batch_idx, channel_idx].mean():.4f}"
    )
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('frame_average_training.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    visualize_frame_average()
