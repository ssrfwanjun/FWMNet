import os
import torch
import numpy as np
import cv2
import time
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from collections import OrderedDict

from model.FWMNet import FWMNet
from transform.data_RGB import get_training_data
from utils.image_utils import torchPSNR, torchSSIM
import utils.model_utils as model_utils
from model.MIRNet_model import MIRNet
from enlighten_inference import EnlightenOnnxModel

def load_model(model_path, device):
    """加载训练好的模型"""
    model = MIRNet()
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except:
            # 处理DataParallel包装的模型
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove 'module.'
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file {model_path} not found. Using random weights.")
    
    model = model.to(device)
    model.eval()
    
    return model

def equalize(input_image):
    out = (input_image - torch.min(input_image))/(torch.max(input_image) - torch.min(input_image))
    return(out)
def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    psnr_values = []
    ssim_values = []
    processing_times = []
    
    # 原始图像与标签图像的对比指标
    input_psnr_values = []
    input_ssim_values = []
    
    print("Evaluating model performance...")
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, desc="Processing")):
            target = data[0].to(device)  # ground truth
            input_ = data[1].to(device)  # input image
            
            # 记录处理时间
            start_time = time.time()
            
            # 模型推理
            restored = model(input_)
            
            # GAN eval
            #restored = np.random.rand(4, 3, 1024, 1024)
            #model = EnlightenOnnxModel(providers = ["CPUExecutionProvider"])
            #restored[0,:,:,:] = model.predict(input_[0,:,:,:].cpu().detach().numpy().transpose(1, 2, 0)).transpose(2, 0, 1)
            #restored[1,:,:,:] = model.predict(input_[1,:,:,:].cpu().detach().numpy().transpose(1, 2, 0)).transpose(2, 0, 1)
            #if i<7:
            #    restored[2,:,:,:] = model.predict(input_[2,:,:,:].cpu().detach().numpy().transpose(1, 2, 0)).transpose(2, 0, 1)
            #    restored[3,:,:,:] = model.predict(input_[3,:,:,:].cpu().detach().numpy().transpose(1, 2, 0)).transpose(2, 0, 1)
            #restored = torch.from_numpy(restored/255).to(device).to(torch.float32)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # 计算PSNR和SSIM - 处理后图像与标签对比
            for res, tar in zip(restored, target):
                psnr_val = torchPSNR(res, tar).item()
                ssim_val = torchSSIM(res, tar).item()
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
            
            # 计算原始输入图像与标签图像的对比
            for inp, tar in zip(input_, target):
                #tmp = cv2.GaussianBlur(equalize(inp).cpu().numpy().transpose(1, 2, 0), (11, 11), 7, 7)
                #inp = torch.from_numpy(tmp.transpose(2, 0, 1)).float().to('cuda')
                #inp = equalize(inp)
                #tar = equalize(tar)
                tmp = inp/torch.mean(inp)*torch.mean(tar)
                tmp = cv2.GaussianBlur(tmp.cpu().numpy().transpose(1, 2, 0), (11,11), 7, 7)
                #tmp = inp*2
                inp = torch.from_numpy(tmp.transpose(2, 0, 1)).float().to('cuda')
                #tmp = cv2.GaussianBlur(tmp.cpu().numpy().transpose(1, 2, 0), (11,11), 7, 7)
                #inp = torch.from_numpy(tmp.transpose(2, 0, 1)).float().to('cuda')
                #tmp = cv2.GaussianBlur(inp.cpu().numpy().transpose(1, 2, 0), (11,11), 7, 7)
                #inp = torch.from_numpy(tmp.transpose(2, 0, 1)).float().to('cuda')*2
                input_psnr_val = torchPSNR(inp, tar).item()
                input_ssim_val = torchSSIM(inp, tar).item()
                input_psnr_values.append(input_psnr_val)
                input_ssim_values.append(input_ssim_val)
    
    return psnr_values, ssim_values, processing_times, input_psnr_values, input_ssim_values


def save_comparison_images(model, data_loader, device, save_dir, num_samples=5):
    """保存对比图像"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving {num_samples} comparison images to {save_dir}")
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_samples:
                break
                
            target = data[0].to(device)  # ground truth
            input_ = data[1].to(device)  # input image
            filename = data[2][0] if len(data) > 2 else f"sample_{i}"
            
            # 模型推理
            restored = model(input_)
            
            # 转换为numpy格式并保存
            input_np = (input_[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            target_np = (target[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            restored_np = (restored[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # 保存单张图像
            cv2.imwrite(os.path.join(save_dir, f"{filename}_input.png"), cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_dir, f"{filename}_target.png"), cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_dir, f"{filename}_restored.png"), cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR))
            
            # 创建四列对比图：输入 vs 处理后 vs 标签 vs 差异图
            h, w = input_np.shape[:2]
            
            # 计算差异图
            input_diff = np.abs(input_np.astype(np.float32) - target_np.astype(np.float32))
            restored_diff = np.abs(restored_np.astype(np.float32) - target_np.astype(np.float32))
            
            # 归一化差异图到0-255
            input_diff = np.clip(input_diff * 2, 0, 255).astype(np.uint8)
            restored_diff = np.clip(restored_diff * 2, 0, 255).astype(np.uint8)
            
            # 创建四列对比图
            comparison_4col = np.hstack([input_np, restored_np, target_np, input_diff])
            cv2.imwrite(os.path.join(save_dir, f"{filename}_comparison_4col.png"), cv2.cvtColor(comparison_4col, cv2.COLOR_RGB2BGR))
            
            # 创建处理前后对比
            comparison_before_after = np.hstack([input_np, restored_np])
            cv2.imwrite(os.path.join(save_dir, f"{filename}_before_after.png"), cv2.cvtColor(comparison_before_after, cv2.COLOR_RGB2BGR))
            
            # 创建完整的对比图（带差异）
            comparison_full = np.hstack([
                input_np, 
                restored_np, 
                target_np,
                input_diff,
                restored_diff
            ])
            cv2.imwrite(os.path.join(save_dir, f"{filename}_full_comparison.png"), cv2.cvtColor(comparison_full, cv2.COLOR_RGB2BGR))
            
            # 保存文本信息
            input_psnr = torchPSNR(input_[0], target[0]).item()
            restored_psnr = torchPSNR(restored[0], target[0]).item()
            input_ssim = torchSSIM(input_[0], target[0]).item()
            restored_ssim = torchSSIM(restored[0], target[0]).item()
            
            info_text = f"""Sample: {filename}
Input PSNR: {input_psnr:.4f} dB, SSIM: {input_ssim:.6f}
Restored PSNR: {restored_psnr:.4f} dB, SSIM: {restored_ssim:.6f}
Improvement: PSNR +{restored_psnr - input_psnr:.4f} dB, SSIM +{restored_ssim - input_ssim:.6f}
"""
            with open(os.path.join(save_dir, f"{filename}_metrics.txt"), 'w') as f:
                f.write(info_text)


def main():
    parser = argparse.ArgumentParser(description='Evaluate FWMNet-64-4-dwt1-512 model on training dataset')
    parser.add_argument('--model_path', type=str, 
                       default='./third_party/MIRNet/checkpoints/model_latest.pth',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, 
                       default='./datasets/traindata/test',
                       help='Path to the training dataset')
    parser.add_argument('--patch_size', type=int, default=1024,
                       help='Patch size for evaluation')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, 
                       default='./evaluation_results/FWMNet-32-4-dwt1-patch512',
                       help='Directory to save results')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--save_images', action='store_true',
                       help='Save comparison images')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of sample images to save')
    
    args = parser.parse_args()
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.model_path, device)
    
    # 准备数据加载器
    print(f"Loading dataset from {args.data_dir}")
    train_dataset = get_training_data(args.data_dir, {'patch_size': args.patch_size})
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # 不打乱，便于追踪
        num_workers=0,  # Windows兼容性
        drop_last=False,
    )
    
    print(f"Dataset loaded: {len(train_dataset)} samples")
    
    # 评估模型
    psnr_values, ssim_values, processing_times, input_psnr_values, input_ssim_values = evaluate_model(model, train_loader, device)
    
    # 计算统计结果 - 处理后图像
    psnr_mean = np.mean(psnr_values)
    psnr_std = np.std(psnr_values)
    psnr_min = np.min(psnr_values)
    psnr_max = np.max(psnr_values)
    
    ssim_mean = np.mean(ssim_values)
    ssim_std = np.std(ssim_values)
    ssim_min = np.min(ssim_values)
    ssim_max = np.max(ssim_values)
    
    # 计算统计结果 - 原始图像
    input_psnr_mean = np.mean(input_psnr_values)
    input_psnr_std = np.std(input_psnr_values)
    input_ssim_mean = np.mean(input_ssim_values)
    input_ssim_std = np.std(input_ssim_values)
    
    # 计算改进
    psnr_improvement = psnr_mean - input_psnr_mean
    ssim_improvement = ssim_mean - input_ssim_mean
    
    avg_time = np.mean(processing_times)
    
    # 打印结果
    print("\n" + "="*80)
    print("EVALUATION RESULTS FOR FWMNet")
    print("="*80)
    print(f"Dataset: {args.data_dir}")
    print(f"Number of samples: {len(psnr_values)}")
    print(f"Patch size: {args.patch_size}")
    print(f"Model: {args.model_path}")
    print("="*80)
    
    # 原始图像指标
    print("ORIGINAL INPUT IMAGES:")
    print("-"*40)
    print(f"  PSNR: {input_psnr_mean:.4f} ± {input_psnr_std:.4f} dB")
    print(f"  SSIM: {input_ssim_mean:.6f} ± {input_ssim_std:.6f}")
    
    print("\nPROCESSED IMAGES (Model Output):")
    print("-"*40)
    print(f"  PSNR: {psnr_mean:.4f} ± {psnr_std:.4f} dB")
    print(f"  SSIM: {ssim_mean:.6f} ± {ssim_std:.6f}")
    print(f"  PSNR Range: [{psnr_min:.4f}, {psnr_max:.4f}] dB")
    print(f"  SSIM Range: [{ssim_min:.6f}, {ssim_max:.6f}]")
    
    print("\nIMPROVEMENT:")
    print("-"*40)
    print(f"  PSNR Improvement: {psnr_improvement:+.4f} dB ({(psnr_improvement/input_psnr_mean*100):+.2f}%)")
    print(f"  SSIM Improvement: {ssim_improvement:+.6f} ({(ssim_improvement/input_ssim_mean*100):+.2f}%)")
    
    print("-"*40)
    print(f"Average processing time: {avg_time:.4f} seconds per image")
    print(f"Total processing time: {np.sum(processing_times):.2f} seconds")
    print("="*80)
    
    # 保存详细结果
    results_file = os.path.join(args.save_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("EVALUATION RESULTS FOR FWMNet-64-4-dwt1-512\n")
        f.write("="*80 + "\n")
        f.write(f"Dataset: {args.data_dir}\n")
        f.write(f"Number of samples: {len(psnr_values)}\n")
        f.write(f"Patch size: {args.patch_size}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write("="*80 + "\n")
        
        f.write("ORIGINAL INPUT IMAGES:\n")
        f.write("-"*40 + "\n")
        f.write(f"  PSNR: {input_psnr_mean:.4f} ± {input_psnr_std:.4f} dB\n")
        f.write(f"  SSIM: {input_ssim_mean:.6f} ± {input_ssim_std:.6f}\n")
        
        f.write("\nPROCESSED IMAGES (Model Output):\n")
        f.write("-"*40 + "\n")
        f.write(f"  PSNR: {psnr_mean:.4f} ± {psnr_std:.4f} dB\n")
        f.write(f"  SSIM: {ssim_mean:.6f} ± {ssim_std:.6f}\n")
        f.write(f"  PSNR Range: [{psnr_min:.4f}, {psnr_max:.4f}] dB\n")
        f.write(f"  SSIM Range: [{ssim_min:.6f}, {ssim_max:.6f}]\n")
        
        f.write("\nIMPROVEMENT:\n")
        f.write("-"*40 + "\n")
        f.write(f"  PSNR Improvement: {psnr_improvement:+.4f} dB ({(psnr_improvement/input_psnr_mean*100):+.2f}%)\n")
        f.write(f"  SSIM Improvement: {ssim_improvement:+.6f} ({(ssim_improvement/input_ssim_mean*100):+.2f}%)\n")
        
        f.write("-"*40 + "\n")
        f.write(f"Average processing time: {avg_time:.4f} seconds per image\n")
        f.write(f"Total processing time: {np.sum(processing_times):.2f} seconds\n")
        f.write("="*80 + "\n")
    
    # 保存详细的数值结果
    details_file = os.path.join(args.save_dir, "detailed_results.csv")
    with open(details_file, 'w') as f:
        f.write("Index,Input_PSNR,Restored_PSNR,Input_SSIM,Restored_SSIM,PSNR_Improvement,SSIM_Improvement,ProcessingTime\n")
        for i, (input_psnr, psnr_val, input_ssim, ssim_val, proc_time) in enumerate(zip(input_psnr_values, psnr_values, input_ssim_values, ssim_values, processing_times)):
            psnr_imp = psnr_val - input_psnr
            ssim_imp = ssim_val - input_ssim
            f.write(f"{i},{input_psnr:.6f},{psnr_val:.6f},{input_ssim:.6f},{ssim_val:.6f},{psnr_imp:.6f},{ssim_imp:.6f},{proc_time:.6f}\n")
    
    print(f"\nResults saved to: {args.save_dir}")
    print(f"Summary: {results_file}")
    print(f"Detailed results: {details_file}")
    
    # 保存对比图像
    if args.save_images:
        image_save_dir = os.path.join(args.save_dir, "comparison_images")
        save_comparison_images(model, train_loader, device, image_save_dir, args.num_samples)


if __name__ == "__main__":
    main()
