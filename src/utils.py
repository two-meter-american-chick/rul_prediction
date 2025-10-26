import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler

def load_and_process_data(data_path, sequence_length=30, clip_threshold=125, window_size=5):
    """
    Загружает, предобрабатывает данные и создает последовательности.
    """
    # Задаем имена колонок
    index_names = ['unit_number', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # Загружаем данные из файлов
    train_df = pd.read_csv(f'{data_path}/train_FD001.txt', sep='\s+', header=None, names=col_names)
    test_df = pd.read_csv(f'{data_path}/test_FD001.txt', sep='\s+', header=None, names=col_names)
    rul_df = pd.read_csv(f'{data_path}/RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])
    
    # --- Предобработка Обучающего набора ---
    max_cycles = train_df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    train_df = pd.merge(train_df, max_cycles, on='unit_number', how='left')
    train_df['RUL'] = train_df['max_cycles'] - train_df['time_cycles']
    train_df.drop(columns=['max_cycles'], inplace=True)
    train_df['RUL'] = train_df['RUL'].clip(upper=clip_threshold)

    # --- Feature Engineering ---
    constant_cols = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    cols_to_drop = setting_names + constant_cols
    train_df_filtered = train_df.drop(columns=cols_to_drop)
    test_df_filtered = test_df.drop(columns=cols_to_drop)
    
    sensor_cols_raw = [col for col in train_df_filtered.columns if col.startswith('s_')]
    for col in sensor_cols_raw:
        train_df_filtered[col + '_mean'] = train_df_filtered.groupby('unit_number')[col].rolling(window_size, min_periods=1).mean().reset_index(level=0, drop=True)
        train_df_filtered[col + '_std'] = train_df_filtered.groupby('unit_number')[col].rolling(window_size, min_periods=1).std().reset_index(level=0, drop=True)
        test_df_filtered[col + '_mean'] = test_df_filtered.groupby('unit_number')[col].rolling(window_size, min_periods=1).mean().reset_index(level=0, drop=True)
        test_df_filtered[col + '_std'] = test_df_filtered.groupby('unit_number')[col].rolling(window_size, min_periods=1).std().reset_index(level=0, drop=True)

    train_df_filtered.fillna(0, inplace=True)
    test_df_filtered.fillna(0, inplace=True)
    
    # --- Нормализация ---
    feature_cols = [col for col in train_df_filtered.columns if col not in ['unit_number', 'time_cycles', 'RUL']]
    scaler = RobustScaler()
    train_df_filtered[feature_cols] = scaler.fit_transform(train_df_filtered[feature_cols])
    test_df_filtered[feature_cols] = scaler.transform(test_df_filtered[feature_cols])

    # --- Создание последовательностей ---
    sequences, labels = [], []
    for unit_id in train_df_filtered['unit_number'].unique():
        unit_data = train_df_filtered[train_df_filtered['unit_number'] == unit_id]
        rul_values = unit_data['RUL'].values
        feature_values = unit_data[feature_cols].values

        for i in range(len(unit_data) - sequence_length + 1):
            sequences.append(feature_values[i:i+sequence_length])
            labels.append(rul_values[i+sequence_length-1])

    X_train_full, y_train_full = np.array(sequences), np.array(labels)
    
    print("Данные успешно загружены и обработаны.")
    print(f"Форма обучающего набора (X_train_full): {X_train_full.shape}")
    
    return X_train_full, y_train_full, test_df_filtered, rul_df, scaler, feature_cols

class TurbofanDataset(Dataset):
    """Класс датасета для PyTorch."""
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences).float()
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]