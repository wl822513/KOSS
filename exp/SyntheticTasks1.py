# 在导入 Triton 或其他库之前先加载 VS 编译环境
from exp.selective_copying_mamba_main.init_env import load_visual_studio_env

# 加载 VS 环境
load_visual_studio_env()

import os
import torch
import torch.optim as optim
import logging
import time
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from exp.selective_copying_mamba_main.config import training_config, dataset_config, MambaConfig
from exp.selective_copying_mamba_main.data_generator import generate_dataset, CopyingDataset
from torch.utils.data import DataLoader
import csv
import sys
import torch.nn as nn
import warnings
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")


def save_training_results(model, optimizer, train_losses, Val_Acc, model_name, intensity,
                          base_model_dir='./models', base_loss_dir='./loss', base_metric_dir='./metrics'):
    """
    仅当当前 Val_Acc 高于历史同 intensity 最优 Val_Acc 时，更新 summary.csv 中对应行，
    并保存最新训练损失和模型参数。

    :param model: PyTorch 模型
    :param optimizer: 优化器
    :param train_Acc: 验证指标 train_Acc
    :param Val_Acc: 测试指标 Val_Acc
    :param model_name: 模型名称
    :param intensity: 污染强度（字符串或数字）
    :param base_model_dir: 模型保存根目录
    :param base_loss_dir: 训练损失保存根目录
    :param base_metric_dir: 评价指标保存根目录
    """

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 路径设置
    model_dir = os.path.join(base_model_dir, model_name)
    loss_dir = os.path.join(base_loss_dir, model_name)
    metric_dir = os.path.join(base_metric_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    # summary.csv 路径
    summary_path = os.path.join(metric_dir, 'summary.csv')

    # 读取 summary.csv，加载所有历史指标
    summary_data = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row['intensity']
                    summary_data[key] = row
        except Exception as e:
            print(f"⚠️ 读取 summary.csv 失败，继续默认空数据: {e}")

    intensity_str = str(intensity)

    # 比较当前 Val_Acc 与历史同强度的 Val_Acc，默认0
    best_val_acc = float(summary_data.get(intensity_str, {}).get('Val_Acc', 0.0))

    if Val_Acc > best_val_acc:
        print(f"🔹 当前 Val_Acc ({Val_Acc:.6f}) 优于历史 {best_val_acc:.6f}，更新保存数据...")

        # 更新 summary_data
        summary_data[intensity_str] = {
            'Val_Acc': f"{Val_Acc:.6f}",
            'intensity': intensity_str,
            'timestamp': current_time
        }

        # 保存 summary.csv（排序写入，方便查看）
        with open(summary_path, 'w', newline='') as f:
            fieldnames = ['intensity', 'Val_Acc', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # 按数值大小排序
            for key in sorted(summary_data.keys(), key=lambda x: float(x)):
                writer.writerow({
                    'intensity': key,
                    'Val_Acc': summary_data[key]['Val_Acc'],
                    'timestamp': summary_data[key].get('timestamp', '')
                })

        print(f"✅ summary.csv 已更新: {summary_path}")

        # 保存训练损失（覆盖）
        loss_path = os.path.join(loss_dir, f'loss_di{intensity_str}.csv')
        with open(loss_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Loss', 'intensity'])
            for epoch, loss in enumerate(train_losses, 1):
                writer.writerow([epoch, loss, intensity_str])
        print(f"✅ 训练损失已保存: {loss_path}")

        # 保存模型参数（覆盖）
        model_path = os.path.join(model_dir, f'model_di{intensity_str}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'Val_Acc': Val_Acc,
            'intensity': intensity_str,
            'timestamp': current_time
        }, model_path)
        print(f"✅ 模型已保存: {model_path}")

    else:
        print(f"❌ 当前 Val_Acc ({Val_Acc:.6f}) 未优于历史 {best_val_acc:.6f}，跳过保存。")


# Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger()

# # Device configuration
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# logger.info(f'Using device: {device}')

# Define model
# mambaconfig = MambaConfig()
# model = MambaLMHeadModel(mambaconfig, device=device)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])

# **添加 models 目录到 sys.path**
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前 train.py 所在目录 (exp)
project_root = os.path.dirname(current_dir)  # 获取 KF-O3S1 目录
models_path = os.path.join(project_root, "models")  # 确保 models 目录在 sys.path 里
if models_path not in sys.path:
    sys.path.append(models_path)


def model_selective(modelSelection, device):
    """根据 modelSelection 选择并实例化不同的模型"""

    # **模型字典：映射模型名称到 (模块名, 参数变量, 模型类)**
    model_dict = {
        "LSTM": ("LSTMDefinite", "paramsLSTM", "LSTMModel"),
        "Mamba": ("MambaDefiniteSt", "paramsMamba", "MambaModel"),
        "Transformer": ("TransformerDefine", "paramsTransformer", "TransformerModel"),
        "KOSS": ("KOSSDefiniteSt", "paramsKOSS", "KOSSModel"),
    }

    # 检查是否是有效模型
    if modelSelection not in model_dict:
        raise ValueError(f"❌ 未知的模型选择: {modelSelection}")

    # 获取模块、参数、模型名称
    module_name, params_name, model_name = model_dict[modelSelection]

    # 动态导入模块
    module = __import__(module_name, fromlist=[params_name, model_name])

    # 获取参数和模型
    params = getattr(module, params_name)
    ModelClass = getattr(module, model_name)

    # 通用参数解析
    model_kwargs = {
        "input_features": params["input_features"],
        "output_features": params["output_features"],
    }

    # 各模型的额外参数
    if modelSelection == "LSTM":
        model_kwargs.update({
            "lstm_hidden_size": params["lstm_hidden_size"],
            "lstm_num_layers": params["lstm_num_layers"],
            "output_time_steps": params["output_time_steps"],
            "input_time_steps": params["input_time_steps"],
            "period": params["period"],
        })
    elif modelSelection == "Mamba":
        model_kwargs.update({
            "mamba_hidden_size": params["mamba_hidden_size"],
            "mamba_num_layers": params["mamba_num_layers"],
            "output_time_steps": params["output_time_steps"],
            "period": params["period"],
        })
    elif modelSelection == "Transformer":
        model_kwargs.update({
            "transformer_hidden_size": params["transformer_hidden_size"],
            "num_transformer_layers": params["num_transformer_layers"],
            "output_time_steps": params["output_time_steps"],
            "num_heads": params["num_heads"],
            "input_time_steps": params["input_time_steps"],
        })
    elif modelSelection == "KOSS":
        model_kwargs.update({
            "KOSS_hidden_size": params["KOSS_hidden_size"],
            "KOSS_num_layers": params["KOSS_num_layers"],
        })

    # 实例化模型并移动到设备
    model = ModelClass(**model_kwargs).to(device)

    print(f"✅ 已选择模型: {modelSelection}")
    return model


def setup_logging():
    """
    设置日志格式和级别。
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    return logging.getLogger()


def get_device():
    """
    获取当前使用设备（GPU 或 CPU）。
    """
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def build_model(modelSelection, device):
    """
    初始化模型与优化器。
    返回：model, model_hyperparams（字典，可选）
    """
    if modelSelection == "Mamba":
        config = MambaConfig()
        model = MambaLMHeadModel(config, device=device)
    elif modelSelection == "KOSS":
        model = model_selective(modelSelection, device)

    else:
        raise ValueError(f"Unsupported model selection: {modelSelection}")

    return model


def prepare_dataloader():
    """
    构建 DataLoader，用于训练。
    """
    train_dataset = CopyingDataset(dataset_config, training_config)
    num_workers = min(8, os.cpu_count())
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader


def train_one_page(model, optimizer, criterion, device, logger, Page):
    """
    训练一个 page（相当于一个 epoch），包含多个 step（每 step 是一个 batch）。
    """
    model.train()
    train_dataset = CopyingDataset(dataset_config, training_config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)  # batch_size=1 是因为 __getitem__ 返回的是 (batch_size, seq_len)
    steps_per_page = len(train_dataset)

    for step, (inputs, targets) in enumerate(train_loader):
        # inputs, targets: (1, batch_size, seq_len or l_memorize)
        inputs = inputs.squeeze(0).to(device)   # shape: (batch_size, seq_len)
        targets = targets.squeeze(0).to(device)  # shape: (batch_size, l_memorize)

        outputs = model(inputs, num_last_tokens=dataset_config['l_memorize']).logits
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            correct = (outputs.argmax(1) == targets).sum().item()
            total = targets.numel()
            accuracy = 100 * correct / total
            progress = 100 * (Page + 1) / training_config["num_page"]

        logger.info(
            f'Page: {Page + 1}/{training_config["num_page"]} ({progress:.2f}%) | '
            f'Step [{step+1}/{steps_per_page}] | '
            f'Loss: {loss.item():.4f} | '
            f'Accuracy: {accuracy:.2f}%'
        )
    return loss.item()


# def train(model, optimizer, criterion, device, logger):
#     """
#     执行整个训练过程。
#     """
#     logger.info(f"Training on device: {device}")
#     start_time = time.time()
#     total = 0
#
#     for page in range(training_config["num_page"]):
#         logger.info(f"--- Page {page + 1}/{training_config['num_page']} ---")
#         total += train_one_page(model, optimizer, criterion, device, logger, page)
#
#     end_time = time.time()
#     logger.info(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")
#     return total/training_config["num_page"]
def train(model, optimizer, criterion, device, logger):
    """
    执行整个训练过程。
    """
    model.train()
    logger.info(f"Training on device: {device}")
    start_time = time.time()
    train_losses = []  # 用于存储每次训练的损失值

    for page in range(training_config["num_page"]):
        logger.info(f"--- Page {page + 1}/{training_config['num_page']} ---")
        loss = train_one_page(model, optimizer, criterion, device, logger, page)
        train_losses.append(loss)  # 将每次的损失值添加到列表中
        logger.info(f"Page {page + 1} loss: {loss:.4f}")

    end_time = time.time()
    logger.info(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")
    return train_losses  # 返回损失列表


def validate(model, device, logger):
    """
    验证模型性能。
    """
    model.eval()
    with torch.no_grad():
        inputs, targets = generate_dataset(dataset_config, training_config)
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs, num_last_tokens=dataset_config['l_memorize']).logits
        correct = (outputs.argmax(1) == targets).sum().item()
        total = targets.size(0) * targets.size(1)
        accuracy = 100 * correct / total
        logger.info(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy


# 选择模型
# modelSelection = "LSTM"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"
modelSelection = "Mamba"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"
# modelSelection = "Transformer"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"
# modelSelection = "KOSS"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"


def main():
    """
    主函数：初始化组件并执行训练与验证。
    """
    logger = setup_logging()
    device = get_device()
    model = build_model(modelSelection, device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    loss = train(model, optimizer, criterion, device, logger)
    accuracy = validate(model, device, logger)

    base_model_dir = './return/models'
    base_loss_dir = './return/loss'
    base_metric_dir = './return/metrics'
    save_training_results(model, optimizer, loss, accuracy, modelSelection, dataset_config["intensity"],
                          base_model_dir, base_loss_dir, base_metric_dir)


if __name__ == '__main__':
    main()




