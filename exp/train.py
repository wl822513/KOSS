"""
日志：
20240530: 用标准的建模架构
20240530：要先将经纬度这种绝对数据转换为dis-ang这种相对数据，但训练要用经纬度数据里避免0-360°翻转的问题
20240531: 引入加性注意力机制，效果非常好
20240606: LSTM+transformer+KF+自注意力 效果非常好
20240615: 改成固定周期,重写lstm_model_test
20240615: 改写了各种注意力模型
20240615: 改写成informer(效果很好)
20240618: test里融合预测值和观测值
20240618: test里融合预测值和观测值dis和ang分开融合
"""
import torch
import numpy as np
import csv
import torch.nn as nn
from tqdm import tqdm
from functions import inverse_standardize, cartesian_to_polar, haversine_distances
from matplot_func import polar_plot_actual_pred1, plot_training_losses
# 现在可以导入 src 里的模块
from dataloaders.prepare.ssr.data_loader import load_data, batch_data_loaders, prepare_test_data

import pandas as pd
import os
import warnings
import matlab.engine

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")


# 获取当前脚本所在的目录路径
current_folder = os.path.dirname(os.path.abspath(__file__))
ssr_kf_dll_path = os.path.join(current_folder, 'SSR_KF_dll')
# 启动 MATLAB 引擎
eng = matlab.engine.start_matlab()
# 添加 SSR_KF_dll 文件夹到 MATLAB 路径
eng.addpath(ssr_kf_dll_path, nargout=0)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        早停类，用于监控验证损失
        Args:
            patience (int): 没有改善的验证损失允许的最大次数
            min_delta (float): 最小改善值，如果变化小于这个值，则不认为有改善
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class MultiStepMultiFeatureLoss1(nn.Module):
    def __init__(self, output_time_steps, output_features, weight=None,
                 scaling_factor=20, first_order_weight=0.3, second_order_weight=0.3):
        super(MultiStepMultiFeatureLoss1, self).__init__()
        if weight is not None:
            assert weight.shape == (output_time_steps, output_features), (
                f"权重的形状应为 ({output_time_steps}, {output_features})"
            )
            self.weight = nn.Parameter(weight)
        else:
            self.weight = None
        self.scaling_factor = scaling_factor
        self.first_order_weight = first_order_weight
        self.second_order_weight = second_order_weight

    def forward(self, y_pred_filtered, y_true):
        assert y_pred_filtered.shape == y_true.shape, "预测和目标的形状必须相同"

        # 计算主误差项：平方误差的平均值
        mse_term = torch.mean((y_true - y_pred_filtered) ** 2)

        # 如果 output_time_steps >= 2，计算一阶和二阶差分惩罚项
        if y_true.size(1) >= 2:
            first_order_pred = y_pred_filtered[:, 1:, :] - y_pred_filtered[:, :-1, :]
            first_order_true = y_true[:, 1:, :] - y_true[:, :-1, :]
            # 一阶差分惩罚项
            penalty_term = torch.mean((first_order_pred - first_order_true) ** 2)
        else:
            # 如果时间步小于 2，则无法计算一阶和二阶差分，不包含惩罚项
            penalty_term = 0.0

        # 自适应的 lambda_value
        scaled_penalty_term = self.scaling_factor * penalty_term
        log_term = torch.log1p(scaled_penalty_term)  # log1p(x) = log(1 + x)，避免对0的直接对数
        lambda_value = torch.sigmoid(log_term)

        # 最终总损失 = 主误差项 + 动态权重的惩罚项
        total_loss = mse_term + lambda_value * penalty_term

        # 应用权重（如果有的话）
        if self.weight is not None:
            weighted_loss = 0.0
            for t in range(y_true.size(1)):  # 遍历时间步
                for f in range(y_true.size(2)):  # 遍历特征
                    weighted_loss += self.weight[t, f] * nn.functional.mse_loss(y_pred_filtered[:, t, f], y_true[:, t, f])
            total_loss += weighted_loss

        return total_loss


def train_model(model, train_loader, criterion, optimizer, scheduler, input_features, output_features,
                num_epochs, device, min_lr=1e-6, patience=5):
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0003)
    train_losses = []
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs[:, :, 1:input_features+1])  # 传递进去flag StateVector
            loss = criterion(outputs, targets[:, :, 2:output_features+2])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'Batch Loss': f'{loss.item():.6e}'})
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        pbar.close()

        # 学习率调度器更新并确保学习率不低于最小值
        scheduler.step()
        for param_group in optimizer.param_groups:
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr

        # 检查早停
        early_stopping(avg_epoch_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_losses


def save_model(model, optimizer, num_epochs, train_losses, model_nmae, base_model_dir='./models'):

    # 设置模型路径，包含当前文件名
    # 确保模型保存的目录和train_losses子目录存在
    final_directory = os.path.join(base_model_dir, 'modelSavePth')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    model_path = os.path.join(final_directory, f'{model_nmae}.pth')

    # 保存模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_losses,
    }, model_path)
    print(f"模型已保存到 {model_path}")


def save_training_losses(train_loss, model_nmae, base_model_dir='./loss'):

    # 设置保存训练损失的CSV文件名，将 'train_losses_' 添加到文件名前面
    csv_file_name = f"loss_{model_nmae}.csv"

    # 确保模型保存的目录和train_losses子目录存在
    final_directory = os.path.join(base_model_dir, 'train_loss')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # 生成完整的CSV文件路径
    csv_file_path = os.path.join(final_directory, csv_file_name)

    # 将训练损失保存到CSV文件
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Loss'])
        for epoch, loss in enumerate(train_loss, 1):
            writer.writerow([epoch, loss])

    print(f"Train losses have been saved to {csv_file_path}")


def predict(model, X_test_tensor):
    model.eval()
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)
        y_pred = predictions_tensor.detach().cpu().numpy()
    return y_pred


def predict1(model, X_test_tensor):
    # 在 predict() 的基础上额外添加一个全 0 维度，并返回新的 y_pred_with_zero。
    model.eval()
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)

        y_pred = predictions_tensor.detach().cpu().numpy()
        # 获取原始数组的形状
        num_samples, num_timesteps, num_features = y_pred.shape
        # 创建一个全为0的新列
        new_feature_column = np.zeros((num_samples, num_timesteps, 1))

        # 将新列与原始数据合并
        y_pred_with_zero = np.concatenate((new_feature_column, y_pred), axis=2)
    return y_pred_with_zero


def lstm_model_test_1(test_data_norm, min_max_test, model, input_time_steps, input_features, period, time_mean, time_std,
                      device, timescale=0.15, alpha=1):
    # 遍历测试数据集并进行预测
    fusion_steps_num = 0  # 融合的时间步
    model.eval()
    with torch.no_grad():
        predictions = []
        predictions_arrays = []  # 用于存储所有 predictions_array
        sequence_array = np.empty((1, 2))
        sequence_array_temp = np.empty((1, 3))
        i = 0
        while i < test_data_norm.shape[0]:
            if i == input_time_steps - 1:
                sequence_array = test_data_norm[:input_time_steps, 1:input_features+2]
                X_test = np.expand_dims(sequence_array, axis=0)
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                predictions_array = predict1(model, X_test_tensor)
                # 确保 predictions_data 是一个二维数组
                predictions_array = predictions_array.squeeze()

                # 融合观测值和预测值
                predictions_data = predictions_array[fusion_steps_num, :]
                observed_values = test_data_norm[input_time_steps-1, 1:input_features+2]
                fused_values = alpha * predictions_data + (1 - alpha) * observed_values

                time = test_data_norm[input_time_steps-1, 0]  # 不管怎么预测的都是下一个时间步，下一个时间步的时间就是+period
                time = np.expand_dims(np.array([time]), axis=1)

                aa = 1
                aa = np.expand_dims(np.array([aa]), axis=1)
                # 扩展 fused_values 的维度
                fused_values = fused_values.reshape(1, -1)
                new_row = np.concatenate((time, fused_values, aa), axis=1)
                predictions.append(new_row)
                predictions_arrays.append(predictions_array)  # 调试用
                i += 1

            elif i >= input_time_steps:
                temp = predictions[-1][0]
                time_interval = test_data_norm[i, 0] - temp[0]
                if time_interval <= 3*period/2 - timescale:  # 2个周期内
                    in_start = i-input_time_steps+1
                    sequence_array = test_data_norm[in_start:i+1, 1:input_features+2]
                    X_test = np.expand_dims(sequence_array, axis=0)
                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                    predictions_array = predict1(model, X_test_tensor)

                    # 确保 predictions_data 是一个二维数组
                    predictions_array = predictions_array.squeeze()

                    # 融合观测值和预测值
                    predictions_data = predictions_array[fusion_steps_num, :]
                    observed_values = test_data_norm[input_time_steps - 1, 1:input_features + 2]
                    fused_values = alpha * predictions_data + (1 - alpha) * observed_values

                    time = test_data_norm[i, 0]  # 不管怎么预测的都是下一个时间步，下一个时间步的时间就是+period
                    time = np.expand_dims(np.array([time]), axis=1)

                    aa = 2
                    aa = np.expand_dims(np.array([aa]), axis=1)
                    # 扩展 fused_values 的维度
                    fused_values = fused_values.reshape(1, -1)
                    new_row = np.concatenate((time, fused_values, aa), axis=1)
                    predictions.append(new_row)
                    predictions_arrays.append(predictions_array)
                    i += 1

                else:
                    temp = predictions[-1][0]
                    new_row = predictions_array[-1]
                    sequence_array = sequence_array[1:, :]  # 去掉首行
                    sequence_array = np.insert(sequence_array, sequence_array.shape[0], new_row, axis=0)  # 插入到最后一行

                    X_test = np.expand_dims(sequence_array, axis=0)
                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                    predictions_array = predict1(model, X_test_tensor)

                    # 确保 predictions_data 是一个二维数组
                    predictions_array = predictions_array.squeeze()

                    # 融合观测值和预测值
                    predictions_data = predictions_array[fusion_steps_num, :]
                    observed_values = new_row
                    fused_values = alpha * predictions_data + (1 - alpha) * observed_values

                    time = temp[0] + period
                    time = np.expand_dims(np.array([time]), axis=1)

                    aa = 3
                    aa = np.expand_dims(np.array([aa]), axis=1)
                    # 扩展 fused_values 的维度
                    fused_values = fused_values.reshape(1, -1)
                    new_row = np.concatenate((time, fused_values, aa), axis=1)
                    predictions.append(new_row)
                    predictions_arrays.append(predictions_array)
                    continue

            else:
                i += 1


        predictions = np.array(predictions)
        predictions_2d = predictions.reshape(-1, predictions.shape[-1])

        predictions_arrays = np.array(predictions_arrays)
        predictions_arrays = predictions_arrays.reshape(-1, predictions_arrays.shape[-1])
        col2 = inverse_standardize(predictions_arrays[:, 3], min_max_test[0, 0], min_max_test[0, 1])
        col3 = inverse_standardize(predictions_arrays[:, 4], min_max_test[1, 0], min_max_test[1, 1])
        predicted_denorm_col1, predicted_denorm_col2 = cartesian_to_polar(col2, col3)
        predictions_arrays = np.column_stack((predicted_denorm_col1, predicted_denorm_col2))

        col2 = inverse_standardize(sequence_array_temp[:, 1], min_max_test[0, 0], min_max_test[0, 1])
        col3 = inverse_standardize(sequence_array_temp[:, 2], min_max_test[1, 0], min_max_test[1, 1])
        predicted_denorm_col1, predicted_denorm_col2 = cartesian_to_polar(col2, col3)
        sequence_array_temp = np.column_stack((sequence_array_temp[:, 0], predicted_denorm_col1, predicted_denorm_col2))
        # 定义保存路径
        save_path = '../result'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 定义保存文件名
        file_name = 'sequence_array_temp.csv'
        file_name1 = 'predictions_arrays.csv'
        # 将数组保存为 CSV 文件
        file_path = os.path.join(save_path, file_name)
        pd.DataFrame(sequence_array_temp).to_csv(file_path, index=False, header=False)
        file_path = os.path.join(save_path, file_name1)
        pd.DataFrame(predictions_arrays).to_csv(file_path, index=False, header=False)

        predictions_2d[:, 3] = inverse_standardize(predictions_2d[:, 3], min_max_test[0, 0], min_max_test[0, 1])
        predictions_2d[:, 4] = inverse_standardize(predictions_2d[:, 4], min_max_test[1, 0], min_max_test[1, 1])
        # 选择 predictions_2d 的第1列、第3列和第4列
        predictions_2d_reordered = predictions_2d[:, [0, 3, 4, 7]]
        return predictions_2d_reordered


def lstm_model_test_2(test_data_norm, min_max_test, model, input_time_steps, input_features, period, time_mean, time_std,
                      device, timescale=0.15):
    model.eval()
    with (torch.no_grad()):
        predictions = []
        i = 0
        while i < test_data_norm.shape[0]:
            if i == input_time_steps - 1:
                sequence_array = test_data_norm[:input_time_steps, 1:input_features+1]
                X_test_tensor = torch.tensor(np.expand_dims(sequence_array, axis=0), dtype=torch.float32).to(device)
                predictions_array = predict1(model, X_test_tensor)

                # 不管怎么预测的都是下一个时间步，下一个时间步的时间就是+period
                batch_size, time_steps, features = predictions_array.shape
                time = np.expand_dims(np.repeat(np.array([test_data_norm[input_time_steps - 1, 0]]), time_steps),
                                      axis=1).reshape(batch_size, time_steps, -1)

                aa = np.expand_dims(np.repeat(1, time_steps), axis=1).reshape(batch_size, time_steps, -1)  # (1, 2, 1)

                # 将时间、预测数据、标志项按列组合
                time_norm = (time - time_mean) / time_std
                new_row = np.concatenate([time, time_norm, predictions_array, aa], axis=2)

                predictions = new_row.copy()  # 创建 predictions 的副本
                i += 1

            elif i >= input_time_steps:
                temp = predictions[-1, -1, 0]
                time_interval = test_data_norm[i, 0] - temp
                in_start = i - input_time_steps + 1
                sequence_array = test_data_norm[in_start:i + 1, 1:input_features+1]
                distances = haversine_distances(sequence_array[:, 1:4])
                dd = time_interval <= 3*period/2 + timescale
                gg = distances[-1] < 5.5*(distances[0]+distances[1])

                if dd and gg:  # 2个周期内
                    sequence_tensor = torch.tensor(np.expand_dims(sequence_array, axis=0), dtype=torch.float32).to(device)
                    X_test_tensor = torch.cat((X_test_tensor, sequence_tensor), dim=0)  # 现在的形状是 (2, 4, 5)
                    predictions_array = predict1(model, X_test_tensor)[-1, :, :][None, :, :]

                    # 不管怎么预测的都是下一个时间步，下一个时间步的时间就是+period
                    batch_size, time_steps, features = predictions_array.shape
                    time = np.expand_dims(np.repeat(np.array([test_data_norm[i, 0]]), time_steps),
                                          axis=1).reshape(batch_size, time_steps, -1)

                    aa = np.expand_dims(np.repeat(2, time_steps), axis=1).reshape(batch_size, time_steps,
                                                                                  -1)  # (1, 2, 1)
                    # 将时间、预测数据、标志项按列组合
                    time_norm = (time - time_mean) / time_std
                    new_row = np.concatenate([time, time_norm, predictions_array, aa], axis=2)
                    predictions = np.concatenate([predictions, new_row], axis=0)
                    i += 1

                else:
                    new_row = predictions[-1, -1, :][None, :]
                    sequence_array = np.concatenate([sequence_array[1:, :], new_row[:, 1:input_features+1]], axis=0)
                    sequence_tensor = torch.tensor(np.expand_dims(sequence_array, axis=0), dtype=torch.float32).to(device)
                    X_test_tensor = torch.cat((X_test_tensor, sequence_tensor), dim=0)  # 现在的形状是 (2, 4, 5)
                    predictions_array = predict1(model, X_test_tensor)[-1, :, :][None, :, :]

                    # 不管怎么预测的都是下一个时间步，下一个时间步的时间就是+period
                    batch_size, time_steps, features = predictions_array.shape
                    time = np.expand_dims(np.repeat(np.array([new_row[0, 0] + period]), time_steps),
                                          axis=1).reshape(batch_size, time_steps, -1)

                    aa = np.expand_dims(np.repeat(3, time_steps), axis=1).reshape(batch_size, time_steps,
                                                                                  -1)  # (1, 2, 1)
                    # 将时间、预测数据、标志项按列组合
                    time_norm = (time - time_mean) / time_std
                    new_row = np.concatenate([time, time_norm, predictions_array, aa], axis=2)
                    if dd:
                        test_data_norm[i] = new_row[0, 0, :-1]
                    else:
                        test_data_norm = np.insert(test_data_norm, i, new_row[0, 0, :-1], axis=0)
                    predictions = np.concatenate([predictions, new_row], axis=0)
                    i += 1
                    continue

            else:
                i += 1

        predictions_2d = predictions[:, 0, :]

        predictions_arrays = predictions.reshape(-1, predictions.shape[-1])
        col2 = inverse_standardize(predictions_arrays[:, 3], min_max_test[0, 0], min_max_test[0, 1])
        col3 = inverse_standardize(predictions_arrays[:, 4], min_max_test[1, 0], min_max_test[1, 1])
        predicted_denorm_col1, predicted_denorm_col2 = cartesian_to_polar(col2, col3)
        predictions_arrays = np.column_stack((predicted_denorm_col1, predicted_denorm_col2))
        # 定义保存路径
        save_path = 'dataset/result'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 定义保存文件名
        file_name1 = 'predictions_arrays.csv'
        file_path = os.path.join(save_path, file_name1)
        pd.DataFrame(predictions_arrays).to_csv(file_path, index=False, header=False)

        predictions_2d[:, 3] = inverse_standardize(predictions_2d[:, 3], min_max_test[0, 0], min_max_test[0, 1])
        predictions_2d[:, 4] = inverse_standardize(predictions_2d[:, 4], min_max_test[1, 0], min_max_test[1, 1])
        # 选择 predictions_2d 的第1列、第3列和第4列
        predictions_2d_reordered = predictions_2d[:, [0, 3, 4, 7]]
        return predictions_2d_reordered


def dataset_selection(datasetSelection, params, snr_db):
    """根据 datasetSelection 选择并加载相应的数据集"""

    # 数据集路径字典
    dataset_dict = {
        "ssr": {
            "train_path": os.path.join("..", "ssr_dataset", "train_dataset", "processed_csv5_0915"),
            "test_path": os.path.join("..", "ssr_dataset", "test_dataset", "A39_source", "2332.csv"),
            "vis_path": os.path.join("..", "vision")
        }
    }

    # 检查数据集是否有效
    if datasetSelection not in dataset_dict:
        raise ValueError(f"❌ 未知的数据集选择: {datasetSelection}")

    # 获取数据集路径
    dataset_paths = dataset_dict[datasetSelection]
    train_path, test_path, vis_path = dataset_paths["train_path"], dataset_paths["test_path"], dataset_paths["vis_path"]

    # 训练集处理
    X_trains, y_trains = load_data(train_path, params["input_time_steps"], params["output_time_steps"], snr_db)
    train_loader = batch_data_loaders(X_trains, y_trains, params["batch_size"])

    print(f"✅ 已加载数据集: {datasetSelection}")
    return vis_path, train_loader


def model_eva(model, params, device):
    if datasetSelection == "ssr":
        test_path = r'../ssr_dataset/test_dataset/A39_source/2332.csv'
        # vis_path = '../vision'
        # 测试
        test_data_norm, min_max_test, test_data, time_mean, time_std = prepare_test_data(test_path, params["input_features"])
        predictions_2d = lstm_model_test_1(test_data_norm, min_max_test, model, params["input_time_steps"],
                                           params["input_features"], params["period"], time_mean, time_std, device)
        predicted_denorm_col1, predicted_denorm_col2 = cartesian_to_polar(predictions_2d[:, 1], predictions_2d[:, 2])
        # 画图
        polar_plot_actual_pred1(test_data['dis'].to_numpy(), test_data['ang'].to_numpy(),
                                predicted_denorm_col1, predicted_denorm_col2)


import sys

# **添加 models 目录到 sys.path**
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前 train.py 所在目录 (exp)
project_root = os.path.dirname(current_dir)  # 获取 KOSS1 目录
models_path = os.path.join(project_root, "models")  # 确保 models 目录在 sys.path 里
if models_path not in sys.path:
    sys.path.append(models_path)


def model_selective(modelSelection):
    """根据 modelSelection 选择并实例化不同的模型"""
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # **模型字典：映射模型名称到 (模块名, 参数变量, 模型类)**
    model_dict = {
        "LSTM": ("LSTMDefinite", "paramsLSTM", "LSTMModel"),
        "Mamba": ("MambaDefinite", "paramsMamba", "MambaModel"),
        "Transformer": ("TransformerDefine", "paramsTransformer", "TransformerModel"),
        "KOSS": ("KOSSDefinite", "paramsKOSS", "KOSSModel"),
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
        "output_time_steps": params["output_time_steps"],
        "output_features": params["output_features"],
    }

    # 各模型的额外参数
    if modelSelection == "LSTM":
        model_kwargs.update({
            "lstm_hidden_size": params["lstm_hidden_size"],
            "lstm_num_layers": params["lstm_num_layers"],
            "input_time_steps": params["input_time_steps"],
            "period": params["period"],
        })
    elif modelSelection == "Mamba":
        model_kwargs.update({
            "mamba_hidden_size": params["mamba_hidden_size"],
            "mamba_num_layers": params["mamba_num_layers"],
            "period": params["period"],
        })
    elif modelSelection == "Transformer":
        model_kwargs.update({
            "transformer_hidden_size": params["transformer_hidden_size"],
            "num_transformer_layers": params["num_transformer_layers"],
            "num_heads": params["num_heads"],
            "input_time_steps": params["input_time_steps"],
        })
    elif modelSelection == "KOSS":
        model_kwargs.update({
            "KOSS_hidden_size": params["KOSS_hidden_size"],
            "KOSS_num_layers": params["KOSS_num_layers"],
            "period": params["period"]
        })

    # 实例化模型并移动到设备
    model = ModelClass(**model_kwargs).to(device)

    print(f"✅ 已选择模型: {modelSelection}")
    return params, model, device


# 选择模型
modelSelection = "LSTM"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"
# modelSelection = "Mamba"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"
# modelSelection = "Transformer"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"
# modelSelection = "KOSS"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"
datasetSelection = "ssr"  # 二次雷达数据
snr_db = 45


def main():
    params, model, device = model_selective(modelSelection)
    vis_path, train_loader = dataset_selection(datasetSelection, params, snr_db)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = MultiStepMultiFeatureLoss1(params["output_time_steps"], params["output_features"],
                                           weight=None).to(device)
    # 训练
    train_losses = train_model(model, train_loader, criterion, optimizer, scheduler, params["input_features"],
                               params["output_features"], params["num_epochs"], device)
    # 画损失函数
    plot_training_losses(train_losses)
    save_training_losses(train_losses, modelSelection, vis_path)
    save_model(model, optimizer,  params["num_epochs"], train_loader, modelSelection, vis_path)
    model_eva(model, params, device)


if __name__ == "__main__":
    main()
