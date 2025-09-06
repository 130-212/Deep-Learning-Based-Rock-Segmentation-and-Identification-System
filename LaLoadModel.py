import torch  # 导入PyTorch库
import timm  # 导入timm库，用于创建预训练模型
from segment_anything import sam_model_registry  # 导入segment_anything库中的模型注册表
import sys
import os

# 设置设备为GPU（如果可用），否则为CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resource_path(relative_path):
    """ 获取资源的绝对路径。用于PyInstaller打包后定位资源文件。"""
    try:
        if hasattr(sys, '_MEIPASS'):
            # 打包后运行，sys._MEIPASS是临时文件夹的路径
            base_path = sys._MEIPASS
        else:
            # 开发环境下，使用当前脚本所在目录
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        # 打印调试信息
        print(f"Base path: {base_path}")
        print(f"Relative path: {relative_path}")
        
        full_path = os.path.join(base_path, relative_path)
        print(f"Full path: {full_path}")
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            print(f"文件不存在: {full_path}")
            # 尝试其他可能的路径
            alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)
            print(f"尝试备用路径: {alt_path}")
            if os.path.exists(alt_path):
                return alt_path
            else:
                print(f"备用路径也不存在: {alt_path}")
                return full_path  # 仍然返回原始路径，让调用者处理错误
        
        return full_path
    except Exception as e:
        print(f"resource_path 错误: {str(e)}")
        # 返回原始路径
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

def load_model():
    # 定义模型权重文件路径列表 - 使用 resource_path 包装
    model_paths = [
        resource_path('models/mobilenetv3_large.pth'),  # MobileNetV3 Large
        resource_path('models/inception_v3.pth'),       # Inception V3
        resource_path('models/deit_small.pth'),         # DeiT Small
        resource_path('models/deit_base.pth')           # DeiT Base
    ]
    
    # 定义模型名称列表 - 使用与权重文件兼容的模型名称
    model_names = [
        'mobilenetv3_large_100',    # 与 mobilenetv3_large.pth 兼容
        'inception_v3',             # 与 inception_v3.pth 兼容
        'deit_small_patch16_224',   # 与 deit_small.pth 兼容
        'deit_base_patch16_224'     # 与 deit_base.pth 兼容
    ]
    
    # 定义每个模型的类别数 - 根据您的实际任务调整
    num_classes = [5, 5, 5, 5]  # 假设都是5分类任务
    
    models = []  # 用于存放加载的模型

    # 遍历模型名称、权重路径和类别数
    for model_name, ckpt_path, num_class in zip(model_names, model_paths, num_classes):
        print(f'正在加载 {model_name}...')  # 打印正在加载的模型名称
        print(f'模型路径: {ckpt_path}')     # 打印模型路径
        
        try:
            # 检查模型文件是否存在
            if not os.path.exists(ckpt_path):
                print(f"模型文件不存在: {ckpt_path}")
                models.append(None)
                continue
                
            # 创建模型，不加载预训练权重，指定类别数
            model = timm.create_model(model_name, pretrained=False, num_classes=num_class)
            
            # 加载模型权重
            state_dict = torch.load(ckpt_path, map_location=device)
            
            # 检查并处理权重键不匹配的问题
            model_state_dict = model.state_dict()
            
            # 筛选出与模型架构匹配的权重
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"跳过不匹配的键: {k}")
            
            # 加载筛选后的权重
            model_state_dict.update(filtered_state_dict)
            model.load_state_dict(model_state_dict)
            
            # 将模型移动到指定设备
            model.to(device)
            # 设置为评估模式
            model.eval()
            # 添加到模型列表
            models.append(model)
            print(f"{model_name} 加载成功!")
            
        except Exception as e:
            print(f"加载 {model_name} 时出错: {str(e)}")
            # 如果加载失败，添加None作为占位符
            models.append(None)

    return models  # 返回所有加载的模型

mymodels = load_model()  # 加载所有模型

def load_SAM():
    print('load SAM ing...')  # 打印正在加载SAM模型

    sam_checkpoint = resource_path("models/sam_vit_h_4b8939.pth") # SAM模型权重路径
    print(f"SAM模型路径: {sam_checkpoint}")  # 打印SAM模型路径
    
    # 检查SAM模型文件是否存在
    if not os.path.exists(sam_checkpoint):
        print(f"SAM模型文件不存在: {sam_checkpoint}")
        # 尝试其他可能的路径
        alt_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sam_vit_h_4b8939.pth"),
            os.path.join(os.getcwd(), "models", "sam_vit_h_4b8939.pth"),
            "models/sam_vit_h_4b8939.pth"
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                sam_checkpoint = alt_path
                print(f"使用备用路径: {sam_checkpoint}")
                break
        else:
            # 如果所有路径都不存在，抛出异常
            raise FileNotFoundError(f"找不到SAM模型文件。尝试的路径: {alt_paths}")

    model_type = "vit_h"  # SAM模型类型

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 再次设置设备

    # 从注册表中获取SAM模型并加载权重
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # 将SAM模型移动到指定设备
    sam.to(device=device)

    return sam  # 返回SAM模型

sam = load_SAM()  # 加载SAM模型