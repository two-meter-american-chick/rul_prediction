import torch
import torch.nn as nn

class RUL_CNN_LSTM_Attention_v6(nn.Module):
    """
    Гибридная модель CNN-LSTM с механизмом Attention для прогнозирования RUL.
    """
    def __init__(self, input_size, dropout_prob):
        super(RUL_CNN_LSTM_Attention_v6, self).__init__()

        # Сверточный блок
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # LSTM блок
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True, dropout=dropout_prob)

        # Attention
        self.attn_w = nn.Linear(128 * 2, 128 * 2) # *2, так как LSTM двунаправленный
        self.attn_u = nn.Linear(128 * 2, 1, bias=False)

        # Полносвязный блок
        self.fc1 = nn.Linear(128 * 2, 128)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.maxpool(out)

        out = out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)

        u = torch.tanh(self.attn_w(lstm_out))
        attn_scores = self.attn_u(u).squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)

        context_vector = torch.sum(lstm_out * attn_weights, dim=1)

        out = self.dropout(self.relu(self.fc1(context_vector)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.fc3(out)

        return out

class AsymmetricMSELoss(nn.Module):
    """
    Асимметричная MSE, которая сильнее штрафует за переоценку RUL.
    """
    def __init__(self, overestimation_penalty=2.5):
        super().__init__()
        self.overestimation_penalty = overestimation_penalty

    def forward(self, y_pred, y_true):
        errors = y_pred - y_true
        penalty_weights = torch.ones_like(errors)
        penalty_weights[errors > 0] = self.overestimation_penalty
        loss = torch.mean(penalty_weights * (errors ** 2))
        return loss