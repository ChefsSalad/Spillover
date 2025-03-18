#使用GRU——GAN，使用短周期的7天技术指标，窗口选择为10
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from model import VAE,Generator_transformer,Discriminator3,sliding_window
import math
import os
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy

#美股使用2490行，A股使用2400行
data = pd.read_csv(r'E:\pythonProgram\spillover\database\gold\gc_daily_with_pct_change.csv', index_col = 'date')

#分为特征和标签
#将 Close 列（收盘价）复制到一个新列 y 中，作为目标值。y 代表模型需要预测的变量。
data['y'] = data[1:]['Close']


# 数据1和数据2是固定的，数据3到数据6需要随机组合
data1_and_2_columns = list(range(0, 6))  # 数据1和数据2的列索引
# 数据3到数据6的列索引
data_columns_options = [
    [],
    [],
    list(range(6,12)),  # 数据3
    list(range(12, 18)),# 数据4
    list(range(18, 24)) # 数据5
    #list(range(26, 32))
      # 数据6
]

columns_to_use_part1 = data1_and_2_columns+data_columns_options[0]+data_columns_options[1]


# 提取两部分数据
x1 = data.iloc[1:, columns_to_use_part1].values


print("Type of x1:", type(x1))
# 检查形状是否一致（仅作为参考）
print("Shape of data:", data.shape)
print("Shape of x1:", x1.shape)

# 提取 'y' 列（即目标值）作为输出,这个是一致的
y = data['y'].values
# 检查输入特征和目标值的形状
print(f"Input features shape: {x1.shape}")
print(f"Target values shape: {y.shape}")

#将数据分成三部分，70%,15%和15%。
train_split = int(data.shape[0] * 0.7)
val_split = int(data.shape[0] * 0.85)

# 两个GAN的训练数据不一样，测试数据也不一样，但是y值是一致的
train_x1, val_x1, test_x1 = x1[:train_split, :], x1[train_split:val_split, :], x1[val_split:, :]
train_y, val_y, test_y = y[:train_split, ], y[train_split:val_split, ], y[val_split:, ]
#使用MinMaxScaler对x和y进行归一化处理，使数据的值分布在[0, 1]范围内或者【-1,1】
#数据有负值的话，建议修改成-1到1区间，判别器架构中ReLU, sigmoid更适合0到1区间
print(f'trainX: {train_x1.shape} trainY: {train_y.shape}')
print(f'testX: {test_x1.shape} testY: {test_y.shape}')

#使用MinMaxScaler对x和y进行归一化处理，使数据的值分布在[0, 1]范围内
x_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))

train_x1 = x_scaler.fit_transform(train_x1)
test_x1 = x_scaler.transform(test_x1)
val_x1 = x_scaler.transform(val_x1)


train_y = y_scaler.fit_transform(train_y.reshape(-1, 1))
test_y = y_scaler.transform(test_y.reshape(-1, 1))
val_y= y_scaler.transform(val_y.reshape(-1, 1))
#使用二元交叉熵（Binary Cross Entropy, BCE）来计算重建误差。
#计算KL散度（Kullback-Leibler Divergence），约束潜在变量分布接近标准正态分布。

#两个GAN具有不同的train_loader
train_loader1 = DataLoader(TensorDataset(torch.from_numpy(train_x1).float()), batch_size = 128, shuffle = False)
#需要和输入的特征数对应
#两个GAN，这里重复进行两次

do_VAE=False
if do_VAE:
    model1 = VAE([18, 400, 400, 400, 10], 10)   

    use_cuda = 1
    device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
    num_epochs = 300
    learning_rate = 0.00003
    model1 = model1.to(device)   
    optimizer1 = torch.optim.Adam(model1.parameters(), lr = learning_rate)

    hist = np.zeros(num_epochs) 
    for epoch in range(num_epochs):
        total_loss = 0
        loss_ = []
        for (x, ) in train_loader1:
            x = x.to(device)
            output, z, mu, logVar = model1(x)
            kl_divergence = 0.5* torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(output, x) + kl_divergence
            loss.backward()
            optimizer1.step()
            loss_.append(loss.item())
        hist[epoch] = sum(loss_)
        print('[{}/{}] Loss:'.format(epoch+1, num_epochs), sum(loss_))

    plt.figure(figsize=(12, 6))
    plt.plot(hist)

    model1.eval()
    _, VAE_train_x1, train_x_mu1, train_x_var1 = model1(torch.from_numpy(train_x1).float().to(device))
    _, VAE_test_x1, test_x_mu1, test_x_var1 = model1(torch.from_numpy(test_x1).float().to(device))
    _,VAE_val_x1,val_x_mu1,val_x_var1=model1(torch.from_numpy(val_x1).float().to(device))


#使用的是x_和y_gan_，x_是用于训练的数据，y_gan的作用有两个，分成两部分，前windows行和win+1行
#第一部分是作为真实数据和生成器的假数据混合，得到训练判别器的假数据
#第二部分是最后一行，是被预测的那一天的真实值

    train_x1 = np.concatenate((train_x1, VAE_train_x1.cpu().detach().numpy()), axis = 1)
    test_x1 = np.concatenate((test_x1, VAE_test_x1.cpu().detach().numpy()), axis = 1)
    val_x1=np.concatenate((val_x1,VAE_val_x1.cpu().detach().numpy()),axis=1)


window_size=3
#窗口大小，可以设置成3,5,10,20等数据，修改完之后，需要修改D中的第一个卷积的输入，需要修改
train_x_slide1, train_y_slide1, train_y_gan1 = sliding_window(train_x1, train_y, window_size)
test_x_slide1, test_y_slide1, test_y_gan1 = sliding_window(test_x1, test_y, window_size)
val_x_slide1,val_y_slide1,val_y_gan1=sliding_window(val_x1,val_y,window_size)



print(f'train_x: {train_x_slide1.shape} train_y: {train_y_slide1.shape} train_y_gan: {train_y_gan1.shape}')
print(f'test_x: {test_x_slide1.shape} test_y: {test_y_slide1.shape} test_y_gan: {test_y_gan1.shape}')
print(f'val_x: {val_x_slide1.shape} val_y: {val_y_slide1.shape} val_y_gan: {val_y_gan1.shape}')

use_cuda = 1
device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")

batch_size = 64
#learning_rate = 0.00016
#可能需要调整，从而避免梯度消失或者是其他loss的问题
learning_rate = 0.00004
num_epochs = 300

trainDataloader1 = DataLoader(TensorDataset(train_x_slide1, train_y_gan1), batch_size = batch_size, shuffle = False)
#两个生成器和两个判别器
#这里需要给出inputsize，就是特征的个数+VAE操作之后增加的10个
dim=train_x_slide1.shape[2]
modelG1 = Generator_transformer(dim).to(device)
#二元交叉熵【损失函数，可能会有问题
criterion = nn.BCELoss()
optimizerG1 = torch.optim.Adam(modelG1.parameters(), lr = learning_rate, betas = (0.9, 0.999))

# 参数配置，需要反复调整，从而获得比较理想的实验结果
#也可以根据实际的损失值进行动态调整权重
#用于记录损失值
histG1 = np.zeros(num_epochs)

#用来记录训练过程中的验证集上的损失和分数
hist_val_loss1 = np.zeros(num_epochs)

best_mse1 = float('inf')
best_model_state1 = None
last_model_state1 = None
best_epoch1 = -1


for epoch in range(num_epochs):
    print(f' 第{epoch + 1}轮开始喽')
    lossdata_G1 = []
    lossdata_G2 = []
    lossdata_G3 = []
    lossdata_D1 = []
    lossdata_D2 = []
    lossdata_D3 = []
    for batch_idx, (x1, y1) in enumerate(trainDataloader1):
        x1 = x1.to(device)
        y1 = y1.to(device)

        #训练D
        # G生成的数据
        fake_data_temp_G1 = modelG1(x1)
        #计算生成器1的生成数据和真实数据之间的差异，用MSE！！！
        loss_mse_G1=F.mse_loss(fake_data_temp_G1.squeeze(), y1[:,-1,:].squeeze())
        loss_G1=loss_mse_G1
        
        #保存损失值
        lossdata_G1.append(loss_G1.item()) 
        #根据批次的奇偶性交叉训练两个GAN,更新G1和G2的参数
        modelG1.zero_grad()
        loss_G1.backward()
        optimizerG1.step()

    histG1[epoch] = sum(lossdata_G1) 
    #使用验证集对G1和G2的生成效果进行验证

    #对G1的生成效果进行验证
    modelG1.eval()
    y_val_true1 = y_scaler.inverse_transform(val_y_slide1)#真实值
    pred_y_val1 = modelG1(val_x_slide1.to(device))#预测值
    y_val_pred1= y_scaler.inverse_transform(pred_y_val1.cpu().detach().numpy())#反归一化
    # 计算G1的MSE值
    mse_val_g1 = mean_squared_error(y_val_true1, y_val_pred1)
    hist_val_loss1[epoch] = mse_val_g1
    # 如果当前轮次的MSE是所有轮次中最小的，保留这个生成器的参数
    if mse_val_g1 < best_mse1:
        best_mse1 = mse_val_g1
        #best_model_state1 = modelG1.state_dict()
        # 使用 deepcopy 方法进行深拷贝
        best_model_state1 = copy.deepcopy(modelG1.state_dict())
        best_epoch1 = epoch + 1 
    if epoch == num_epochs - 1:
        last_model_state1 = copy.deepcopy(modelG1.state_dict())
    # 切换回训练模式
    modelG1.train()
  

# 将结果保存在output文件夹中
output_dir = 'output_transformer'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打印本次训练所设置的参数
print(f'Generator 1 features (indices): {columns_to_use_part1}')
print(f'learning_rate:{learning_rate}')
print(f'window_size:{window_size}')
print(f'num_epochs:{num_epochs}')
print(f'batch_size:{batch_size}')
print(f'Best MSE1 achieved at epoch: {best_epoch1}')

print("这是在验证集上表现最好的模型的测试结果：")
#切换到评估模式，得到训练结果和预测结果的曲线，进行融合输出
model_test1 = Generator_transformer(dim).to(device)
model_test1.load_state_dict(best_model_state1)
model_test1.eval()

#训练数据和测试数据的真实值
y_train_true = y_scaler.inverse_transform(train_y_slide1)
y_test_true = y_scaler.inverse_transform(test_y_slide1)
#G1的结果
pred_y_train1 = model_test1(train_x_slide1.to(device))
pred_y_test1 = model_test1(test_x_slide1.to(device))
#对预测值和真实值进行反归一化，将其恢复到原始价格范围。
y_train_pred1 = y_scaler.inverse_transform(pred_y_train1.cpu().detach().numpy())
y_test_pred1 = y_scaler.inverse_transform(pred_y_test1.cpu().detach().numpy())

#G1的生成器训练损失
plt.figure(figsize=(12, 8))
plt.plot(y_train_true, color = 'black', label = 'Acutal Price')
plt.plot(y_train_pred1 , color = 'blue', label = 'Predict Price')
plt.title('gru_gan prediction training dataset')
plt.ylabel('close')
plt.xlabel('days')
plt.legend(loc = 'upper right')
prediction_training_path = os.path.join(output_dir, 'gru_gan_best_val_prediction_training.png')
plt.savefig(prediction_training_path)
# 计算并输出G1预测数据和真实数据的指标
mse_g1_train = mean_squared_error(y_train_true, y_train_pred1)
mae_g1_train = mean_absolute_error(y_train_true, y_train_pred1)
rmse_g1_train = np.sqrt(mse_g1_train)
mape_g1_train = np.mean(np.abs((y_train_true - y_train_pred1) / y_train_true)) * 100
print(f'gru_gan best_val Training MSE: {mse_g1_train}')
print(f'gru_gan best_val Training MAE: {mae_g1_train}')
print(f'gru_gan best_val Training RMSE: {rmse_g1_train}')
print(f'gru_gan best_val Training MAPE: {mape_g1_train}%')

#G1的测试数据
plt.figure(figsize=(12, 8))
plt.plot(y_test_true, color = 'black', label = 'Acutal Price')
plt.plot(y_test_pred1 , color = 'blue', label = 'Predict Price')
plt.title('gru_gan prediction testing dataset')
plt.ylabel('close')
plt.xlabel('days')
plt.legend(loc = 'upper right')
prediction_training_path = os.path.join(output_dir, 'gru_gan_best_val_prediction_testing.png')
plt.savefig(prediction_training_path)
# 计算并输出G1测试数据和真实数据的指标
mse_g1_test = mean_squared_error(y_test_true, y_test_pred1)
mae_g1_test = mean_absolute_error(y_test_true, y_test_pred1)
rmse_g1_test = np.sqrt(mse_g1_test)
mape_g1_test = np.mean(np.abs((y_test_true - y_test_pred1) / y_test_true)) * 100
print(f'gru_gan best_val Testing MSE: {mse_g1_test}')
print(f'gru_gan best_val Testing MAE: {mae_g1_test}')
print(f'gru_gan best_val Testing RMSE: {rmse_g1_test}')
print(f'gru_gan best_val Testing MAPE: {mape_g1_test}%')

#将最后的预测值和真实值保存到csv文件中，一共8组
results_train = {
    'train_best_val_true': y_train_true.flatten(),
    'G1_best_val_train_pred': y_train_pred1.flatten()
}
results_test={
    
    'test_best_val_true': y_test_true.flatten(),
    'G1_test_best_val_pred': y_test_pred1.flatten()
    
}

results_train_df = pd.DataFrame(results_train)
results_train_csv_path = os.path.join(output_dir, 'dataset_best_val__train_results.csv')
results_train_df.to_csv(results_train_csv_path, index=False)
print(f'Results saved to {results_train_csv_path}')

results_test_df = pd.DataFrame(results_test)
results_test_csv_path = os.path.join(output_dir, 'dataset_best_val_test_results.csv')
results_test_df.to_csv(results_test_csv_path, index=False)
print(f'Results saved to {results_test_csv_path}')

#这是在最后保存的模型上的测试结果
print("这是最后一轮保存的模型的测试结果：")
#切换到评估模式，得到训练结果和预测结果的曲线，进行融合输出
model_test2 = Generator_transformer(dim).to(device)
model_test2.load_state_dict(last_model_state1)
model_test2.eval()

#训练数据和测试数据的真实值
y_train_true = y_scaler.inverse_transform(train_y_slide1)
y_test_true = y_scaler.inverse_transform(test_y_slide1)

pred_y_train_b = model_test2(train_x_slide1.to(device))
pred_y_test_b = model_test2(test_x_slide1.to(device))
#对预测值和真实值进行反归一化，将其恢复到原始价格范围。
y_train_pred_b = y_scaler.inverse_transform(pred_y_train_b.cpu().detach().numpy())
y_test_pred_b = y_scaler.inverse_transform(pred_y_test_b.cpu().detach().numpy())

#G1的生成器训练损失
plt.figure(figsize=(12, 8))
plt.plot(y_train_true, color = 'black', label = 'Acutal Price')
plt.plot(y_train_pred_b , color = 'blue', label = 'Predict Price')
plt.title('gru_gan prediction training dataset')
plt.ylabel('close')
plt.xlabel('days')
plt.legend(loc = 'upper right')
prediction_training_path = os.path.join(output_dir, 'gru_gan_last_prediction_training.png')
plt.savefig(prediction_training_path)
# 计算并输出G1预测数据和真实数据的指标
mse_g1_train = mean_squared_error(y_train_true, y_train_pred_b)
mae_g1_train = mean_absolute_error(y_train_true, y_train_pred_b)
rmse_g1_train = np.sqrt(mse_g1_train)
mape_g1_train = np.mean(np.abs((y_train_true - y_train_pred_b) / y_train_true)) * 100
print(f'gru_gan last Training MSE: {mse_g1_train}')
print(f'gru_gan last Training MAE: {mae_g1_train}')
print(f'gru_gan last Training RMSE: {rmse_g1_train}')
print(f'gru_gan last Training MAPE: {mape_g1_train}%')

#G1的测试数据
plt.figure(figsize=(12, 8))
plt.plot(y_test_true, color = 'black', label = 'Acutal Price')
plt.plot(y_test_pred_b , color = 'blue', label = 'Predict Price')
plt.title('gru_gan prediction testing dataset')
plt.ylabel('close')
plt.xlabel('days')
plt.legend(loc = 'upper right')
prediction_training_path = os.path.join(output_dir, 'gru_gan_last_prediction_testing.png')
plt.savefig(prediction_training_path)
# 计算并输出G1测试数据和真实数据的指标
mse_g1_test = mean_squared_error(y_test_true, y_test_pred_b)
mae_g1_test = mean_absolute_error(y_test_true, y_test_pred_b)
rmse_g1_test = np.sqrt(mse_g1_test)
mape_g1_test = np.mean(np.abs((y_test_true - y_test_pred_b) / y_test_true)) * 100
print(f'gru_gan last best_val Testing MSE: {mse_g1_test}')
print(f'gru_gan last best_val Testing MAE: {mae_g1_test}')
print(f'gru_gan last best_val Testing RMSE: {rmse_g1_test}')
print(f'gru_gan last best_val Testing MAPE: {mape_g1_test}%')

#将最后的预测值和真实值保存到csv文件中，一共8组
results_train = {
    'train_last_true': y_train_true.flatten(),
    'G1_last_train_pred': y_train_pred1.flatten()
}
results_test={
    
    'test_last_true': y_test_true.flatten(),
    'G1_last_test_pred': y_test_pred1.flatten()
    
}

results_train_df = pd.DataFrame(results_train)
results_train_csv_path = os.path.join(output_dir, 'dataset_last__train_results.csv')
results_train_df.to_csv(results_train_csv_path, index=False)
print(f'Results saved to {results_train_csv_path}')

results_test_df = pd.DataFrame(results_test)
results_test_csv_path = os.path.join(output_dir, 'dataset_last_test_results.csv')
results_test_df.to_csv(results_test_csv_path, index=False)
print(f'Results saved to {results_test_csv_path}')


# 保存生成器和判别器的参数
generator_discriminator_states = {
    'best_model_state1': best_model_state1,
    'last_model_state1': last_model_state1
}
for key, state in generator_discriminator_states.items():
    state_path = os.path.join(output_dir, f'{key}.pth')
    torch.save(state, state_path)
    print(f'{key} saved to {state_path}')