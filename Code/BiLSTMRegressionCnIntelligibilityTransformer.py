import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchviz import make_dot
from sklearn.preprocessing import StandardScaler
import sys

# 1. 数据加载和预处理
class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe):
        self.scaler = StandardScaler()
        dataframe = dataframe.reset_index(drop=True)
        self.labels = dataframe.iloc[:, 2].astype(float).fillna(0)
        features = dataframe.iloc[:, 6:]
        self.features = self.scaler.fit_transform(features.fillna(0))  # 标准化特征

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        features = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0)
        return features, label

# 2. 定义BiLSTM模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # 输出层大小设置为1，适用于回归任务

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 4. 测试模型
def test_model(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
    print(f'Mean Squared Error: {total_loss / len(loader)}')

if __name__ == "__main__":
    sys.stdout = open('./Record/outputCnIntelligibilityTransformer.txt', 'w')
    dataframe = pd.read_csv('./Dataset/CnData.csv', low_memory=False)

    dataframe = dataframe.drop('merge_key1', axis=1)

    # 划分训练集和测试集
    train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=3407)
    train_dataset = TimeSeriesDataset(train_df)
    test_dataset = TimeSeriesDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)  
    test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False)  

    model = BiLSTM(input_size=16381, hidden_size=256, num_layers=2)

    # 训练模型
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00001)
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        for i, (features, labels) in enumerate(train_loader):
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), './Model/ModelForCnIntelligibilityTransformer.pth')
    
    # 获取一个批处理数据样本
    features, _ = next(iter(train_loader))
    outputs = model(features)
    dot = make_dot(outputs, params=dict(list(model.named_parameters()) + [('features', features)]))
    dot.render('model_visualization_for_noiselevel', format='png')

    test_model(test_loader)
    
    sys.stdout.close()
    
    sys.stdout = sys.__stdout__
