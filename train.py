import os

import torch.nn as nn

from utils.data_processor import DataProcessor
from utils.trainer import Trainer

class Config:
    """配置参数类"""
    def __init__(self):
        # 数据配置
        self.data_path = r'E:\pythonProgram\Spillover\database\gold\gc_first_term.csv'
        self.feature_columns = {
            'fixed': list(range(0, 6)),
            'optional': [[]]
        }
        self.target_col = 'Close'
        self.split_ratios = (0.7, 0.15, 0.15)  # 训练, 验证, 测试
        
        # 模型参数
        self.window_size = 10
        self.batch_size = 128
        self.lr = 3e-5
        self.epochs = 300
        self.use_cuda = True
        self.use_vae = False
        self.vae_latent_dim = 10
        
        # 路径配置
        self.output_dir = 'output_gru_gan'
        os.makedirs(self.output_dir, exist_ok=True)


if __name__ == "__main__":
    # 初始化配置
    config = Config()
    config.epochs = 500  # 修改训练轮数
    config.batch_size = 256  # 修改批次大小

    # 数据预处理
    processor = DataProcessor(config)
    data = processor.process()

    # 模型训练
    trainer = Trainer(config, processor)
    model, history = trainer.train(data)
    
