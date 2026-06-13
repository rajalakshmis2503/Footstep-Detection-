import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as torchfunc

class CNNAnomalyDetector(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 16 * 100, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(torchfunc.relu(self.conv1(x)))  # [B,16,32,200]
        x = self.pool(torchfunc.relu(self.conv2(x)))  # [B,32,16,100]
        x = x.view(x.size(0), -1)
        x = self.dropout(torchfunc.relu(self.fc1(x)))
        return self.fc2(x)

    @staticmethod
    def preprocess_audio(file_path, n_mels=64, max_len=400):
        waveform, sr = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=n_mels)
        mel = mel_spectrogram(waveform).squeeze(0)

        if mel.shape[1] < max_len:
            mel = torchfunc.pad(mel, (0, max_len - mel.shape[1]))
        else:
            mel = mel[:, :max_len]

        return mel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 64, 400]

def predict(file_path):
    input_tensor = CNNAnomalyDetector.preprocess_audio(file_path).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNAnomalyDetector(n_classes=2).to(device)
model.load_state_dict(torch.load("cnn_anomaly_detector_best.pth", map_location=device))
model.eval()

def detect_anomaly(audio_file):
    """
    Detect if the given audio file contains an anomaly
    Returns True if anomaly detected, False if normal
    """
    prediction = predict(audio_file)
    return bool(prediction)  # 1 indicates anomaly, 0 indicates normal

