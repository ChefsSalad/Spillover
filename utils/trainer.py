import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt



class GRUGAN(nn.Module):
    """GRU-based Generator"""
    def __init__(self, input_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])



class Trainer:
    """模型训练器"""
    def __init__(self, config, processor):
        self.config = config
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
        
    def create_loaders(self, data):
        """创建数据加载器"""
        (train_x, val_x, test_x), (train_y, val_y, test_y) = data
        
        # 创建滑动窗口
        def create_windows(x, y, window_size):
            x_slide, y_slide = [], []
            for i in range(len(x) - window_size):
                x_slide.append(x[i:i+window_size])
                y_slide.append(y[i+window_size])
            return np.array(x_slide), np.array(y_slide)
        
        train_x, train_y = create_windows(train_x, train_y, self.config.window_size)
        val_x, val_y = create_windows(val_x, val_y, self.config.window_size)
        test_x, test_y = create_windows(test_x, test_y, self.config.window_size)
        
        # 转换为TensorDataset
        train_ds = TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
        val_ds = TensorDataset(torch.FloatTensor(val_x), torch.FloatTensor(val_y))
        test_ds = TensorDataset(torch.FloatTensor(test_x), torch.FloatTensor(test_y))
        
        return (
            DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=self.config.batch_size),
            DataLoader(test_ds, batch_size=self.config.batch_size)
        )

    def train(self, data):
        """完整训练流程"""
        # 准备数据
        train_loader, val_loader, test_loader = self.create_loaders(data)
        
        # 初始化模型
        input_dim = data[0][0].shape[-1]
        model = GRUGAN(input_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        criterion = nn.MSELoss()
        
        # 训练循环
        best_loss = float('inf')
        history = {'train': [], 'val': []}
        for epoch in range(self.config.epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            history['train'].append(train_loss/len(train_loader))
            
            # 验证阶段
            val_loss = self.evaluate(model, val_loader, criterion)
            history['val'].append(val_loss)
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                
            print(f'Epoch {epoch+1}/{self.config.epochs} | Train Loss: {history["train"][-1]:.4f} | Val Loss: {val_loss:.4f}')
        
        # 最终评估
        model.load_state_dict(best_model)
        test_loss = self.evaluate(model, test_loader, criterion)
        print(f'Final Test Loss: {test_loss:.4f}')
        
        # 保存结果
        self.save_results(model, history, test_loader)
        return model, history

    def evaluate(self, model, loader, criterion):
        """模型评估"""
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                total_loss += criterion(pred, y).item()
        return total_loss / len(loader)

    def save_results(self, model, history, test_loader):
        """保存结果和模型"""
        # 保存模型
        torch.save(model.state_dict(), os.path.join(self.config.output_dir, 'best_model.pth'))
        
        # 保存训练曲线
        plt.plot(history['train'], label='Train Loss')
        plt.plot(history['val'], label='Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.config.output_dir, 'training_curve.png'))
        plt.close()
        
        # 保存测试结果
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                pred = model(x).cpu().numpy()
                preds.extend(pred)
                truths.extend(y.numpy())
        
        # 反归一化
        preds = self.processor.scaler_y.inverse_transform(np.array(preds))
        truths = self.processor.scaler_y.inverse_transform(np.array(truths))
        
        # 保存预测结果
        results = pd.DataFrame({'True': truths.flatten(), 'Predicted': preds.flatten()})
        results.to_csv(os.path.join(self.config.output_dir, 'predictions.csv'), index=False)
        
        # 绘制预测图
        plt.figure(figsize=(12, 6))
        plt.plot(truths, label='True')
        plt.plot(preds, label='Predicted')
        plt.legend()
        plt.savefig(os.path.join(self.config.output_dir, 'predictions.png'))
        plt.close()