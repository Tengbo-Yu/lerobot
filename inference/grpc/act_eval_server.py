import grpc
import torch
import numpy as np
import os
import pickle
import logging
import cv2
from concurrent import futures
import sys
from typing import Dict, Any
from einops import rearrange
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import time

# 添加KL散度计算函数
def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

# 添加正确的路径以导入模块
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))  # 指向项目根目录
inference_dir = os.path.join(parent_dir, "inference", "grpc")
sys.path.append(parent_dir)  # 添加项目根目录以导入ACT模块
sys.path.append(inference_dir)

from proto import policy_pb2
from proto import policy_pb2_grpc

# 从项目根目录导入所需模块
from policy import ACTPolicy
from utils import set_seed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置参数解析器
import argparse
parser = argparse.ArgumentParser(description="ACT Policy Evaluation Server")
parser.add_argument("--ckpt_path", type=str, required=True, help="预训练策略检查点路径")
parser.add_argument("--stats_path", type=str, required=True, help="数据集统计数据的pickle文件路径")
parser.add_argument("--port", type=int, default=50051, help="监听端口")
parser.add_argument("--seed", type=int, default=1000, help="随机种子")
parser.add_argument("--temporal_agg", action="store_true", help="使用时间聚合进行预测")
parser.add_argument("--policy_class", type=str, default="ACT", help="策略类别（ACT或CNNMLP）")
parser.add_argument("--target_resolution", type=str, default="480x640", help="Target resolution for resizing images (HxW)")
args = parser.parse_args()

# 根据平台和可用性确定设备
if torch.cuda.is_available():
    device = "cuda"  
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  
else:
    device = "cpu"  

logger.info(f"使用设备: {device}")

def make_policy(policy_class, policy_config):
    """根据策略类别创建并返回策略模型"""
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError(f"不支持的策略类别: {policy_class}")
    return policy

def make_policy_config(camera_names):
    """为ACT创建策略配置"""
    policy_config = {
        'lr': 1e-4,  # 推理时不使用
        'num_queries': 100,  # 默认查询数量
        'kl_weight': 1.0,  # 推理时不使用
        'hidden_dim': 512,  # 与模型配置匹配
        'dim_feedforward': 3200,  # 与模型配置匹配
        'lr_backbone': 1e-5,  # 推理时不使用
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
        # 添加必需的参数，仅用于模型初始化，推理时不使用
        'ckpt_dir': os.path.dirname(args.ckpt_path),  # 提取检查点目录
        'policy_class': args.policy_class,
        'task_name': 'sim_transfer_cube',  # 默认任务名，推理时不重要
        'seed': args.seed,
        'num_epochs': 1,  # 推理时不使用
        'batch_size': 1,  # 推理时不使用
        'weight_decay': 1e-4,  # 推理时不使用
        'lr_drop': 1,  # 推理时不使用
        'clip_max_norm': 0.1,  # 推理时不使用
        'position_embedding': 'sine',  # 位置编码类型
        'dilation': False,  # 不使用扩张卷积
        'dropout': 0.1,  # dropout率
        'pre_norm': False,  # 不使用pre-norm
        'masks': False,  # 不使用掩码
        'eval': True,  # 推理模式
    }
    return policy_config

def setup_policy(ckpt_path, stats_path, policy_class='ACT', camera_names=['main']):
    """设置策略和预处理函数"""
    # 加载数据集统计信息
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    logger.info(f"从{stats_path}加载数据集统计信息")
    logger.info(f"qpos_mean维度: {stats['qpos_mean'].shape}, qpos_std维度: {stats['qpos_std'].shape}")
    logger.info(f"action_mean维度: {stats['action_mean'].shape}, action_std维度: {stats['action_std'].shape}")
    
    # 配置策略
    policy_config = make_policy_config(camera_names)
    
    # 创建策略
    policy = make_policy(policy_class, policy_config)
    policy.to(device)
    
    # 加载权重
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    logger.info(f"加载状态: {loading_status}")
    policy.eval()
    logger.info(f"从{ckpt_path}加载{policy_class}策略")
    
    # 创建预处理和后处理函数
    def pre_process(s_qpos):
        """预处理函数，可以处理不同维度的输入"""
        qpos_mean = stats['qpos_mean']
        qpos_std = stats['qpos_std']
        
        # 检查输入维度与统计信息维度是否匹配
        if len(s_qpos) != len(qpos_mean):
            logger.warning(f"输入qpos维度 ({len(s_qpos)}) 与统计信息维度 ({len(qpos_mean)}) 不匹配")
            
            # 情况1: 输入维度大于统计信息维度，只使用前N个维度
            if len(s_qpos) > len(qpos_mean):
                logger.warning(f"输入维度较大，将截断为{len(qpos_mean)}维")
                s_qpos = s_qpos[:len(qpos_mean)]
            
            # 情况2: 输入维度小于统计信息维度，用0填充
            else:
                logger.warning(f"输入维度较小，将填充到{len(qpos_mean)}维")
                padded_qpos = np.zeros_like(qpos_mean)
                padded_qpos[:len(s_qpos)] = s_qpos
                s_qpos = padded_qpos
        
        # 执行归一化
        return (s_qpos - qpos_mean) / qpos_std
    
    def post_process(a):
        """将预测动作转换回原始尺度"""
        return a * stats['action_std'] + stats['action_mean']
    
    return policy, pre_process, post_process, stats, policy_config

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        # 完全绕过argparse，直接构建模型
        import sys
        # 添加必要的导入路径
        detr_path = os.path.join(parent_dir, "detr")
        sys.path.append(detr_path)
        
        # 直接导入必要的模块，绕过数据目录检查
        from models.detr_vae_single import build as build_vae_single
        import argparse
        
        # 创建一个Namespace对象代替解析命令行参数
        args = argparse.Namespace()
        
        # 设置所有必需的参数
        for k, v in args_override.items():
            setattr(args, k, v)
        
        # 确保所有必需的参数都有默认值
        if not hasattr(args, 'position_embedding'):
            setattr(args, 'position_embedding', 'sine')
        if not hasattr(args, 'dilation'):
            setattr(args, 'dilation', False)
        if not hasattr(args, 'dropout'):
            setattr(args, 'dropout', 0.1)
        if not hasattr(args, 'pre_norm'):
            setattr(args, 'pre_norm', False)
        if not hasattr(args, 'masks'):
            setattr(args, 'masks', False)
        if not hasattr(args, 'eval'):
            setattr(args, 'eval', True)  # 推理模式
        if not hasattr(args, 'weight_decay'):
            setattr(args, 'weight_decay', 1e-4)  # 默认权重衰减
        
        logger.info(f"使用以下配置构建ACT模型: {args_override}")
        
        # 直接构建模型，绕过build_ACT_model中的数据检查
        try:
            # 直接使用dual-arm模型（7D actions）
            model = build_vae_single(args)
            model.to(device)
            
            # 创建优化器
            param_dicts = [
                {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
            ]
            optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
            
            self.model = model  # CVAE解码器
            self.optimizer = optimizer
            self.kl_weight = args_override.get('kl_weight', 1.0)
            logger.info(f"ACT模型构建成功，KL权重: {self.kl_weight}")
            
        except Exception as e:
            logger.error(f"构建ACT模型时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # 训练时间
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else:  # 推理时间
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # 无动作，从先验采样
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

class ACTEvalServicer(policy_pb2_grpc.PolicyServiceServicer):
    def __init__(self):
        super().__init__()
        # 设置策略和处理函数
        self.policy, self.pre_process, self.post_process, self.stats, self.policy_config = setup_policy(
            args.ckpt_path, 
            args.stats_path,
            policy_class=args.policy_class,
            camera_names=['main']  # 默认使用main相机
        )
        self.temporal_agg = args.temporal_agg
        self.num_queries = self.policy.model.num_queries
        self.query_frequency = self.num_queries if not self.temporal_agg else 1
        
        # 时间聚合的设置
        self.max_timesteps = 200  # 最大回合长度
        self.current_timestep = 0
        
        # 获取qpos维度（从stats的均值维度）
        self.qpos_dim = len(self.stats['qpos_mean'])
        logger.info(f"使用qpos维度: {self.qpos_dim}")
        
        # 使用正确的qpos维度初始化历史记录
        self.qpos_history = torch.zeros((1, self.max_timesteps, self.qpos_dim)).to(device)
        
        if self.temporal_agg:
            # 确保动作维度与qpos一致
            self.action_dim = len(self.stats['action_mean'])
            self.all_time_actions = torch.zeros([1, self.max_timesteps, self.max_timesteps + self.num_queries, self.action_dim]).to(device)
        self.all_actions = None
        
        # 初始化处理
        set_seed(args.seed)
    
    def get_image(self, image):
        """处理图像，使用与eval_bc函数相同的方法"""
        try:
            # 首先记录原始输入图像信息
            logger.info(f"进入get_image，原始图像：形状={image.shape}, 类型={image.dtype}, 通道数={image.shape[2] if len(image.shape) > 2 else 1}")
            
            # 确保图像有3个通道
            if len(image.shape) == 2:  # 灰度图像
                image = np.stack([image, image, image], axis=2)
                logger.info(f"将灰度图像转换为3通道，新形状: {image.shape}")
            elif len(image.shape) == 3 and image.shape[2] == 1:  # 单通道图像
                image = np.concatenate([image, image, image], axis=2)
                logger.info(f"将单通道图像转换为3通道，新形状: {image.shape}")
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA图像
                image = image[:, :, :3]  # 只保留RGB通道
                logger.info(f"将RGBA图像转换为RGB，新形状: {image.shape}")
            elif len(image.shape) == 3 and image.shape[2] != 3:  # 其他非3通道图像
                # 创建新的3通道图像
                logger.warning(f"遇到非标准通道数 {image.shape[2]}，创建新的3通道图像")
                new_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=image.dtype)
                for i in range(min(3, image.shape[2])):
                    new_image[:, :, i] = image[:, :, i]
                image = new_image
            
            # 再次确认图像现在是3通道
            if len(image.shape) != 3 or image.shape[2] != 3:
                logger.error(f"图像转换后仍不是3通道: {image.shape}，创建新的3通道图像")
                height, width = image.shape[:2]
                image = np.ones((height, width, 3), dtype=np.uint8) * 128  # 创建灰色图像
            
            # 转换为float并归一化到[0, 1]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.float32 or image.dtype == np.float64:
                # 如果已经是浮点数，确保范围在0-1之间
                if image.max() > 1.0:
                    image = image / 255.0
            
            # 记录转换后的图像信息
            logger.info(f"归一化后的图像: 形状={image.shape}, 最小值={image.min():.3f}, 最大值={image.max():.3f}, 通道数={image.shape[2]}")
            
            # 重新排列为CHW格式（ACT需要这种格式）
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            
            # 确认通道数正确
            if image.shape[0] != 3:
                logger.error(f"转置后通道数错误: {image.shape}，应为3通道")
                # 修复通道数
                if image.shape[0] == 1:
                    image = np.repeat(image, 3, axis=0)
                else:
                    # 创建新的3通道图像
                    new_image = np.zeros((3, image.shape[1], image.shape[2]), dtype=image.dtype)
                    for i in range(min(3, image.shape[0])):
                        new_image[i] = image[i]
                    image = new_image
            
            # 转换为tensor
            image_tensor = torch.from_numpy(image).float().to(device)
            logger.info(f"转换为Tensor: 形状={image_tensor.shape}")
            
            # 应用预训练模型的标准化变换
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            image_tensor = normalize(image_tensor)
            
            # 最终确认图像是3通道
            if image_tensor.shape[0] != 3:
                logger.error(f"标准化后通道数错误: {image_tensor.shape}，应为3通道")
                # 创建替代的标准化张量
                empty_tensor = torch.zeros((3, image.shape[1], image.shape[2]), device=device)
                return empty_tensor
            
            logger.info(f"最终图像Tensor: 形状={image_tensor.shape}, 设备={image_tensor.device}")
            return image_tensor
            
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 创建一个标准化的空图像tensor
            try:
                h, w = image.shape[:2]
                empty_tensor = torch.zeros((3, h, w), device=device)
                # 应用标准化，使其与正常处理的图像具有相同的统计特性
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                empty_tensor = normalize(empty_tensor)
                return empty_tensor
            except:
                # 如果无法获取原始图像尺寸，创建一个固定尺寸的图像
                logger.error("无法创建与原始图像相同尺寸的替代图像，使用固定尺寸")
                empty_tensor = torch.zeros((3, 480, 640), device=device)
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                empty_tensor = normalize(empty_tensor)
                return empty_tensor
    
    def decode_image(self, encoded_image, image_format=None, height=None, width=None):
        """将压缩的图像数据解码为numpy数组"""
        try:
            # 将字节转换为numpy数组
            nparr = np.frombuffer(encoded_image, np.uint8)
            
            # 解码图像 - 一定要使用彩色模式
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 强制使用彩色模式
            if img is None:
                logger.error(f"无法解码图像，编码数据大小: {len(encoded_image)}字节")
                raise ValueError("无法解码图像数据")
            
            # 确保BGR图像转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 打印解码后的图像形状和通道数
            logger.info(f"解码JPEG后图像形状: {img.shape}, 类型: {img.dtype}, 通道数: {img.shape[2] if len(img.shape) > 2 else 1}")
            
            # 确保图像有3个通道
            if len(img.shape) != 3 or img.shape[2] != 3:
                logger.error(f"解码后图像不是3通道: {img.shape}")
                # 创建一个3通道的空图像
                rgb_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                
                # 如果是灰度图像，复制到所有通道
                if len(img.shape) == 2:
                    rgb_img[:,:,0] = img
                    rgb_img[:,:,1] = img
                    rgb_img[:,:,2] = img
                    logger.info("将灰度图像转换为RGB")
                # 如果是单通道图像
                elif len(img.shape) == 3 and img.shape[2] == 1:
                    rgb_img[:,:,0] = img[:,:,0]
                    rgb_img[:,:,1] = img[:,:,0]
                    rgb_img[:,:,2] = img[:,:,0]
                    logger.info("将单通道图像转换为RGB")
                img = rgb_img
                logger.info(f"转换后的图像形状: {img.shape}")
            
            return img
            
        except Exception as e:
            logger.error(f"解码图像时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 创建一个备选的3通道RGB图像
            empty_img = np.ones((height or 480, width or 640, 3), dtype=np.uint8) * 128  # 灰色图像
            logger.info(f"返回备选图像，形状: {empty_img.shape}")
            return empty_img
    
    def Predict(self, request, context):
        """
        Main prediction function that handles requests from the client and returns model predictions
        """
        try:
            logger.info("Received prediction request")
            
            # Process the image data
            rgb_image = None
            
            # Check if the request has encoded images
            if hasattr(request, 'encoded_image') and request.encoded_image:
                logger.info(f"Processing encoded wrist image, size: {len(request.encoded_image)/1024:.2f}KB")
                rgb_image = self.decode_image(
                    request.encoded_image,
                    getattr(request, 'image_format', "jpeg"), 
                    getattr(request, 'image_height', None), 
                    getattr(request, 'image_width', None)
                )
                logger.info(f"Decoded wrist image: shape={rgb_image.shape if rgb_image is not None else 'None'}")
            else:
                error_msg = "Request does not contain encoded wrist image"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(error_msg)
                return policy_pb2.PredictResponse()
            
            # Process head camera image if available
            head_image = None
            if hasattr(request, 'encoded_image2') and request.encoded_image2:
                logger.info(f"Processing encoded head image, size: {len(request.encoded_image2)/1024:.2f}KB")
                head_image = self.decode_image(
                    request.encoded_image2,
                    getattr(request, 'image_format', "jpeg"),
                    getattr(request, 'image2_height', None),
                    getattr(request, 'image2_width', None)
                )
                logger.info(f"Decoded head image: shape={head_image.shape if head_image is not None else 'None'}")
            
            # Get the state data
            if not hasattr(request, 'state') or not request.state:
                error_msg = "Request does not contain state data"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(error_msg)
                return policy_pb2.PredictResponse()
            
            state_data = np.array(request.state)
            logger.info(f"State data shape: {state_data.shape}")
            
            # Check connection state for temporal aggregation
            if hasattr(self, 'get_conn_lock') and hasattr(self, 'release_conn_lock'):
                self.get_conn_lock()
                conn_time = self.conn_time
                self.release_conn_lock()
                
                # Check if network was restarted, if so reset current state
                if conn_time == self.last_conn_time and self.input_state != 0:
                    input_state = self.input_state
                else:
                    self.last_conn_time = conn_time
                    input_state = 0
                    self.input_state = 0
                    logger.info("Connection reset, state zeroed")
            else:
                logger.info("No connection lock, not managing connection state")
                input_state = self.pre_process(state_data)
                
            # Process the images for model input
            wrist_tensor = self.get_image(rgb_image)
            
            # Ensure the image tensor has correct dimensions
            if wrist_tensor.dim() == 3:  # CHW format
                logger.info(f"Final image tensor: shape={wrist_tensor.shape}, device={wrist_tensor.device}, channels={wrist_tensor.shape[0]}")
                if wrist_tensor.shape[0] != 3:
                    logger.error(f"Image channels error! Expected 3 channels, got {wrist_tensor.shape[0]}, attempting to fix")
                    # Fix the channel dimension
                    if wrist_tensor.shape[0] == 1:
                        logger.info("Expanding single channel tensor to 3 channels")
                        wrist_tensor = wrist_tensor.expand(3, wrist_tensor.shape[1], wrist_tensor.shape[2])
                    else:
                        logger.info(f"Cropping or padding tensor to 3 channels")
                        new_tensor = torch.zeros(3, wrist_tensor.shape[1], wrist_tensor.shape[2], device=wrist_tensor.device)
                        for i in range(min(3, wrist_tensor.shape[0])):
                            new_tensor[i] = wrist_tensor[i]
                        wrist_tensor = new_tensor
                    logger.info(f"Fixed image tensor: shape={wrist_tensor.shape}, channels={wrist_tensor.shape[0]}")
            
            # Add batch dimension if needed
            if wrist_tensor.dim() == 3:
                wrist_tensor = wrist_tensor.unsqueeze(0)
                logger.info(f"Added batch dimension: shape={wrist_tensor.shape}")
            
            # Make sure tensor is on the correct device
            if wrist_tensor.device != device:
                logger.info(f"Moving tensor from {wrist_tensor.device} to {device}")
                wrist_tensor = wrist_tensor.to(device)
            
            # Perform model inference
            logger.info(f"Performing model inference, input shape: {wrist_tensor.shape}")
            start_time = time.perf_counter()
            with torch.no_grad():
                try:
                    # Call policy model with preprocessed qpos and image tensor
                    if isinstance(input_state, np.ndarray):
                        input_state_tensor = torch.from_numpy(input_state).float().to(device)
                    else:
                        input_state_tensor = input_state
                        
                    # Reshape state tensor if needed
                    if input_state_tensor.dim() == 1:
                        input_state_tensor = input_state_tensor.unsqueeze(0)
                    
                    # Call the model
                    action = self.policy(input_state_tensor, wrist_tensor)
                    
                    if isinstance(action, tuple):
                        action_tensor = action[0]
                    else:
                        action_tensor = action
                        
                    # Convert action to numpy
                    action_np = action_tensor.detach().cpu().numpy()
                    
                    # End timing
                    end_time = time.perf_counter()
                    inference_time_ms = (end_time - start_time) * 1000
                    
                    # Log model output
                    if action_np.size > 0:
                        logger.info(f"Predicted action: shape={action_np.shape}, first 5 values={action_np.flatten()[:5]}")
                    else:
                        logger.warning("Model output action array is empty")
                    
                    # Post-process action
                    raw_action = action_tensor.squeeze(0).cpu().numpy()
                    processed_action = self.post_process(raw_action)
                    
                    # Increment timestep
                    if hasattr(self, 'current_timestep'):
                        self.current_timestep += 1
                        if self.current_timestep >= self.max_timesteps:
                            logger.warning("Reached maximum timesteps, resetting counter")
                            self.current_timestep = 0
                    
                    # Log prediction result
                    logger.info(f"Prediction: {processed_action.tolist()}")
                    
                    # Build response
                    response = policy_pb2.PredictResponse(
                        prediction=processed_action.tolist(),
                        inference_time_ms=inference_time_ms
                    )
                    return response
                    
                except Exception as e:
                    error_msg = f"Model inference failed: {str(e)}"
                    logger.error(error_msg)
                    import traceback
                    logger.error(traceback.format_exc())
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(error_msg)
                    return policy_pb2.PredictResponse()
            
        except Exception as e:
            error_msg = f"Exception in prediction process: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return policy_pb2.PredictResponse()
    
    def HealthCheck(self, request, context):
        """处理健康检查请求"""
        return policy_pb2.HealthCheckResponse(status="healthy")
    
    def GetModelInfo(self, request, context):
        """处理模型信息请求"""
        if self.policy is not None:
            return policy_pb2.ModelInfoResponse(
                status="loaded",
                model_path=args.ckpt_path,
                device=device,
                input_features="qpos, image",
                output_features="action",
                message=f"ACT模型已加载, num_queries: {self.num_queries}, temporal_agg: {self.temporal_agg}"
            )
        else:
            return policy_pb2.ModelInfoResponse(
                status="not_loaded",
                model_path="",
                device=device,
                message="模型加载失败"
            )
    
    def Reset(self, request, context):
        """重置内部状态（例如，用于新回合）"""
        self.current_timestep = 0
        if self.temporal_agg:
            self.all_time_actions = torch.zeros([1, self.max_timesteps, self.max_timesteps + self.num_queries, self.action_dim]).to(device)
        self.qpos_history = torch.zeros((1, self.max_timesteps, self.qpos_dim)).to(device)
        self.all_actions = None
        return policy_pb2.ResetResponse(status="reset_successful")

    def SetPredictionEnabled(self, request, context):
        """处理启用/禁用预测请求"""
        # 可用于暂停/恢复预测
        enabled_status = "已启用" if request.enabled else "已禁用"
        return policy_pb2.SetPredictionEnabledResponse(status=f"预测{enabled_status}")

def serve():
    """Start the gRPC server"""
    # Create a gRPC server with 10 workers
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # 50MB
        ]
    )
    
    # Add the ACTEvalServicer to the server
    policy_pb2_grpc.add_PolicyServiceServicer_to_server(
        ACTEvalServicer(), server
    )
    
    # Listen on port 50051 for compatibility with policy client
    port = args.port
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    logger.info(f"ACT Policy Evaluation Server started on port {port}")
    logger.info(f"Using model: {args.ckpt_path}")
    logger.info(f"Using stats file: {args.stats_path}")
    logger.info(f"Temporal aggregation: {args.temporal_agg}")
    
    try:
        # Keep the server running until interrupted
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        server.stop(0)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="ACT Policy Evaluation Server")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to pretrained policy checkpoint")
    parser.add_argument("--stats_path", type=str, required=True, help="Path to dataset statistics pickle file")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on (default: 50051)")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    parser.add_argument("--temporal_agg", action="store_true", help="Use temporal aggregation for prediction")
    parser.add_argument("--policy_class", type=str, default="ACT", help="Policy class (ACT or other)")
    parser.add_argument("--target_resolution", type=str, default="480x640", help="Target resolution for resizing images (HxW)")
    args = parser.parse_args()
    
    # Parse target resolution
    try:
        TARGET_HEIGHT, TARGET_WIDTH = map(int, args.target_resolution.split('x'))
        logger.info(f"Target resolution set to {TARGET_HEIGHT}x{TARGET_WIDTH}")
    except Exception as e:
        logger.warning(f"Invalid resolution format: {args.target_resolution}. Using default 480x640")
        TARGET_HEIGHT, TARGET_WIDTH = 480, 640
    
    serve() 