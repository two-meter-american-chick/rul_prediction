import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import argparse
from models import RUL_CNN_LSTM_Attention_v6, AsymmetricMSELoss

SEQUENCE_LENGTH = 50
CLIP_THRESHOLD = 125
WINDOW_SIZE = 5

# --- Функции подготовки данных (можно вынести в utils.py) ---
def load_and_preprocess_data(data_path='../data/'):
    # ... (здесь весь код из ячеек 4-7 ноутбука LSTM.ipynb)
    # Загрузка, вычисление RUL, создание признаков, нормализация, создание последовательностей
    # Эта функция должна возвращать X_train, X_val, y_train, y_val
    print("Загрузка и предобработка данных...")
    # ... (код опущен для краткости, его нужно скопировать из ноутбука)
    # Примерный возврат
    # return X_train, X_val, y_train, y_val
    pass # Заглушка

class TurbofanDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences).float()
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def main(args):
    # Воспроизводимость
    np.random.seed(42)
    torch.manual_seed(42)

    # Загрузка данных
    # X_train, X_val, y_train, y_val = load_and_preprocess_data()
    # Для примера создадим заглушки. В реальном коде используйте load_and_preprocess_data
    X_train = np.random.rand(14184, 30, 42)
    X_val = np.random.rand(3547, 30, 42)
    y_train = np.random.rand(14184)
    y_val = np.random.rand(3547)
    
    # Создание DataLoader'ов
    train_dataset = TurbofanDataset(X_train, y_train)
    val_dataset = TurbofanDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Инициализация модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SIZE = X_train.shape[2]
    model = RUL_CNN_LSTM_Attention_v6(INPUT_SIZE, args.dropout).to(device)
    
    criterion = AsymmetricMSELoss(overestimation_penalty=2.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Начинаем обучение на устройстве: {device}")
    
    # Цикл обучения
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()

        avg_train_loss = np.sqrt(train_loss / len(train_loader))
        avg_val_loss = np.sqrt(val_loss / len(val_loader))

        print(f"Эпоха {epoch+1}/{args.epochs}, Train RMSE: {avg_train_loss:.4f}, Val RMSE: {avg_val_loss:.4f}")

    # Сохранение модели
    torch.save(model.state_dict(), 'rul_model_final.pth')
    print("Модель сохранена в rul_model_final.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обучение модели для прогнозирования RUL.')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=64, help='Размер батча')
    parser.add_argument('--lr', type=float, default=0.0001, help='Скорость обучения')
    parser.add_argument('--dropout', type=float, default=0.3, help='Вероятность dropout')
    
    args = parser.parse_args()
    main(args)