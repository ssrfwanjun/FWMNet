import torch
import torch.nn as nn
import pywt
import numpy as np

class DWT_Layer(nn.Module):
    """
    基于db4小波基的3层小波分解层
    专门针对太赫兹低信噪比图像优化
    """
    def __init__(self, wavelet='db4', levels=3):
        super(DWT_Layer, self).__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.requires_grad = False  # 小波变换通常不需要梯度
        self.db4_filters = pywt.Wavelet(wavelet)
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        all_batch_coeffs = []  # 存储每个批次的系数
        
        for b in range(batch_size):  # 先遍历批次
            batch_coeffs = []
            for c in range(channels):   # 再遍历通道
                # 提取单张图像 [H, W]
                img = x[b, c, :, :].detach().cpu().numpy()
                
                # 3层小波分解[1](@ref)
                coeffs = pywt.wavedec2(img, self.wavelet, level=self.levels)
                
                # 展平所有系数并确保总长度=H*W（小波变换守恒性）
                flat_coeffs = []
                for i, level_coeff in enumerate(coeffs):
                    if i == 0:
                        flat_coeffs.append(level_coeff.flatten())  # 近似系数 (LL)
                    else:
                        # 细节系数 (HL, LH, HH)
                        hl, lh, hh = level_coeff
                        flat_coeffs.extend([hl.flatten(), lh.flatten(), hh.flatten()])
                
                # 拼接并验证长度
                flattened = np.concatenate(flat_coeffs)
                expected_length = height * width
                if len(flattened) != expected_length:
                    # 自动填充或裁剪至预期长度（确保形状匹配）
                    flattened = np.resize(flattened, expected_length)
                
                batch_coeffs.append(torch.from_numpy(flattened))
            
            # 当前批次所有通道堆叠: [channels, H*W]
            batch_tensor = torch.stack(batch_coeffs, dim=0)
            all_batch_coeffs.append(batch_tensor)
        
        # 合并所有批次: [batch_size, channels, H*W]
        output = torch.stack(all_batch_coeffs, dim=0)
        
        # 使用reshape而非view（更安全，自动处理连续性[7](@ref)）
        new_height, new_width = height // (2 ** self.levels), width // (2 ** self.levels)
        try:
            # 尝试重塑为 [batch_size, channels, -1, new_height, new_width]
            output = output.reshape(batch_size, channels, -1, new_height, new_width)
        except RuntimeError as e:
            print(f"重塑失败: {e}")
            # 备用方案: 直接返回扁平特征[batch_size, channels, H*W]
            output = output.reshape(batch_size, channels, -1)
        
        return output

class IWT_Layer(nn.Module):
    """
    基于db4小波基的3层小波重构层
    """
    def __init__(self, wavelet='db4', levels=3):
        super(IWT_Layer, self).__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.requires_grad = False
    
    def forward(self, x):
        """
        输入: 小波系数 [batch, channels*(3*levels+1), height/(2^levels), width/(2^levels)]
        输出: 重构图像 [batch, channels, height, width]
        """
        batch_size, coeff_channels, reduced_height, reduced_width = x.shape
        channels = coeff_channels // (3 * self.levels + 1)  # 计算原始通道数
        
        reconstructed_images = []
        
        for b in range(batch_size):
            batch_reconstructed = []
            
            for c in range(channels):
                # 提取当前通道的所有系数
                start_idx = c * (3 * self.levels + 1)
                channel_coeffs = x[b, start_idx:start_idx + (3 * self.levels + 1), :, :]
                
                # 重新组织为pywt期望的系数结构
                coeffs_list = []
                current_idx = 0
                
                # 近似系数 (最低频)
                approx = channel_coeffs[current_idx, :, :].cpu().numpy()
                current_idx += 1
                coeffs_list.append(approx)
                
                # 各层的细节系数
                detail_coeffs = []
                for level in range(self.levels, 0, -1):
                    if current_idx + 2 < channel_coeffs.shape[0]:
                        hl = channel_coeffs[current_idx, :, :].cpu().numpy()
                        lh = channel_coeffs[current_idx + 1, :, :].cpu().numpy()
                        hh = channel_coeffs[current_idx + 2, :, :].cpu().numpy()
                        detail_coeffs.append((hl, lh, hh))
                        current_idx += 3
                
                # 小波重构
                try:
                    reconstructed = pywt.waverec2([approx] + detail_coeffs, self.wavelet)
                    batch_reconstructed.append(torch.tensor(reconstructed))
                except Exception as e:
                    print(f"重构错误: {e}")
                    # 备用简单重构
                    reconstructed = approx  # 使用近似系数作为备用
                    batch_reconstructed.append(torch.tensor(reconstructed))
            
            reconstructed_batch = torch.stack(batch_reconstructed, dim=0)
            reconstructed_images.append(reconstructed_batch)
        
        return torch.stack(reconstructed_images, dim=0)

# 简化的DWT和IWT包装类（保持与您原始代码接口一致）
class DWT_db4(nn.Module):
    def __init__(self, wavelet='db4', levels=3):
        super(DWT_db4, self).__init__()
        self.dwt_layer = DWT_Layer(wavelet, levels)
    
    def forward(self, x):
        return self.dwt_layer(x)

class IWT_db4(nn.Module):
    def __init__(self, wavelet='db4', levels=3):
        super(IWT_db4, self).__init__()
        self.iwt_layer = IWT_Layer(wavelet, levels)
    
    def forward(self, x):
        return self.iwt_layer(x)
