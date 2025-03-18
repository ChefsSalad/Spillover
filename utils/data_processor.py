from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random
class DataProcessor:
    """数据预处理类"""
    def __init__(self, config):
        self.config = config
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        
    def load_data(self):
        """加载并处理原始数据"""
        df = pd.read_csv(self.config.data_path, index_col='time')
        df['y'] = df[1:][self.config.target_col]
        return df

    def prepare_features(self, df):
        """特征工程"""
        # 组合固定特征和随机可选特征
        # selected = self.config.feature_columns['fixed'] + \
        #           random.choice(self.config.feature_columns['optional'])
        selected = self.config.feature_columns['fixed']
        return df.iloc[1:, selected].values, df['y'].values

    def split_data(self, x, y):
        """数据分割"""
        n = len(x)
        train_end = int(n * self.config.split_ratios[0])
        val_end = train_end + int(n * self.config.split_ratios[1])
        return (x[:train_end], x[train_end:val_end], x[val_end:]), \
               (y[:train_end], y[train_end:val_end], y[val_end:])

    def process(self):
        """完整数据处理流程"""
        # 数据加载
        df = self.load_data()
        
        # 特征工程
        x, y = self.prepare_features(df)
        
        # 数据分割
        (train_x, val_x, test_x), (train_y, val_y, test_y) = self.split_data(x, y)
        
        # 数据归一化
        train_x = self.scaler_x.fit_transform(train_x)
        val_x = self.scaler_x.transform(val_x)
        test_x = self.scaler_x.transform(test_x)
        
        train_y = self.scaler_y.fit_transform(train_y.reshape(-1, 1))
        val_y = self.scaler_y.transform(val_y.reshape(-1, 1))
        test_y = self.scaler_y.transform(test_y.reshape(-1, 1))
        
        return (train_x, val_x, test_x), (train_y, val_y, test_y)
