"""
BLE Wearable Python Communication
-------------
A python program can send, receive, plot, and log data from NRF52 IOT_Wearable_Project
-------------
Requirements:
1. bleak: https://github.com/hbldh/bleak
2. matplotlib
3. numpy
4. pandas

"""
# Change this to the number of data points you want to collect
# e.g. 25 = 1 second of data at 25Hz
MAX_TRAIN_SIZE = 1000

# Device ID
BLEName = "We-Be 8ac7"

# Path to save the data, please include the last "/"
DATA_PATH = "Logs/data.csv"



import asyncio
import datetime
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bleak import BleakScanner, BleakClient
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import joblib

UART_SERVICE_UUID = "2a3b580a-7263-4249-a647-e0af6f5966ab"
UART_RX_CHAR_UUID = "2a3b580b-7263-4249-a647-e0af6f5966ab"
UART_TX_CHAR_UUID = "2a3b580c-7263-4249-a647-e0af6f5966ab"

USER_NAME = '58' # 51: Wei-8ac7, 52: Chongzhou-9716, 53: Navid-9716， 54: Tina-9716, 55: Kay-9716, 56: Asmita-8ac7, 57: zequan-8ac7, 58: ruoyu-8ac7

MODEL_PATH = f"model_{USER_NAME}.pth"



LOAD_MODEL = False

# All BLE devices have MTU of at least 23. Subtracting 3 bytes overhead, we can
# safely send 20 bytes at a time to any device supporting this service.
UART_SAFE_SIZE = 241

start_time = time.time()
flag_time = datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S')
size = 150
SentCommand = False
total_bytes = 0
time_gap = 0

x_vec = np.linspace(0, 1, size + 1)[0:-1]
y_vec = np.random.randn(len(x_vec))
line1 = []
title = ''

# queue used for tranfering data between data receiver and model
ppg_queue = asyncio.Queue()

# number of entries in the training set
train_set_done = False
is_stop = False

# model-related variables
scaler = StandardScaler()
label_encoder = LabelEncoder()
if LOAD_MODEL:
    scaler = joblib.load(f"scaler_{USER_NAME}.save")   
    label_encoder = joblib.load(f"label_encoder_{USER_NAME}.pkl")

ppg_columns = ['grn_count', 'grn2Cnt', 'irCnt', 'redCnt']
other_features = []
target = 'User_ID'
sequence_length = 100  # Timesteps per sequence
unknown_users = {24}

data_columns = ['timestamp','temperature','GSR','inp_sample_count','grn_count','grn2Cnt','irCnt','redCnt'
                                  ,'accelx','accely','accelz','whrm_suite_curr_opmode','heart_rate_estim'
                                  ,'hr_confidence','rr_interbeat_interval','rr_confidence'
                                  ,'activity_class','r_spo2','spo2_confidence','spo2_estim','spo2_calc_percentage','spo2_low_sign_quality_flag'
                                  ,'spo2_motion_flag','spo2_low_pi_flag','spo2_unreliable_r_flag','spo2_state'
                                  ,'skin_contact_state','walk_steps','run_steps','kcal','cadence','event']

df_live = pd.DataFrame(columns = data_columns)


def create_test_loader(data):
    global scaler
    global label_encoder
    for col in ppg_columns:
        data[col] = bandpass_filter(data[col])

    data[ppg_columns + other_features] = scaler.transform(data[ppg_columns + other_features])

    X, y = [], []
    seq = data.iloc[:sequence_length][ppg_columns + other_features].values
    X.append(seq)
    y.append(str(USER_NAME))
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    if train_set_done or LOAD_MODEL:
        y = label_encoder.transform(y)
    else:
        y = label_encoder.fit_transform(y)
        joblib.dump(label_encoder, f"label_conder_{USER_NAME}.pkl")

    test_dataset = PPGDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader

# -----------------------------
# Inference with Combined Classifier & Anomaly Detector
# -----------------------------
def predict_with_anomaly(model, X, classifier_threshold=0.5, ae_threshold=0.1, device="cpu"):
    """
    For each sample:
      1. Extract latent features.
      2. Compute reconstruction error using the autoencoder.
      3. If reconstruction error > ae_threshold, label as "unknown".
      4. Otherwise, use classifier prediction (optionally also threshold based on softmax).
    """
    global autoencoder

    model.eval()
    autoencoder.eval()
    all_preds = []
    all_confidences = []
    dataset = PPGDataset(X, np.zeros(len(X)))  # dummy labels
    loader = DataLoader(dataset, batch_size=32)
    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)
            # Get latent features from the BiLSTM model
            features = model(batch_X, return_features=True)
            # Compute reconstruction error
            reconstructed = autoencoder(features)
            rec_error = F.mse_loss(reconstructed, features, reduction='none').mean(dim=1)
            # Get classifier outputs
            outputs = model(batch_X)  # shape: (batch_size, num_known_classes)
            probs = torch.softmax(outputs, dim=1)
            max_probs, pred_indices = torch.max(probs, dim=1)
            pred_labels = [label_encoder.inverse_transform([idx.item()])[0] for idx in pred_indices]
            
            # Determine final predictions based on anomaly detection:
            # If rec_error > ae_threshold, we flag the sample as "unknown"
            final_preds = []
            for error, pl, conf in zip(rec_error, pred_labels, max_probs):
                if error.item() > ae_threshold:
                    final_preds.append("unknown")
                else:
                    # Optionally also apply a classifier confidence threshold if desired
                    final_preds.append(pl if conf.item() >= classifier_threshold else "unknown")
            all_preds.extend(final_preds)
            all_confidences.extend(max_probs.cpu().numpy())
    return all_preds, all_confidences


# -----------------------------
# Define an Autoencoder for Anomaly Detection
# -----------------------------
class FeatureAutoencoder(nn.Module):
    def __init__(self, feature_dim, latent_dim=64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# -----------------------------
# Training the Autoencoder on Known Data Latent Features
# -----------------------------
def train_autoencoder(model, data_loader, ae_optimizer, ae_epochs=1, device="cpu"):
    global autoencoder
    ae_criterion = nn.MSELoss()
    autoencoder.train()
    model.eval()  # Freeze the classifier; we're using it only to extract features
    for epoch in range(ae_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            # Extract latent features and detach to prevent gradients flowing into model
            features = model(batch_X, return_features=True).detach()
            reconstructed = autoencoder(features)
            loss = ae_criterion(reconstructed, features)
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
            epoch_loss += loss.item()
        print(f"AE Epoch {epoch+1}/{ae_epochs}, Loss: {epoch_loss/len(data_loader):.4f}")


def bandpass_filter(signal, lowcut=0.5, highcut=12.0, fs=25.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, num_classes=27):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        
        self.attention = nn.Sequential(  # Add attention mechanism
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        attn_weights = self.attention(out)  # (batch, seq_len, 1)
        out = torch.sum(out * attn_weights, dim=1)  # Weighted sum
        return self.fc(out)
    
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=27, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)  # Softmax over timesteps
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, return_features=False):
        # LSTM output: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        # Attention weights: (batch_size, seq_len, 1)
        attn_weights = self.attention(lstm_out)
        # Compute context vector as weighted sum of lstm_out
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        if return_features:
            return context  # Return latent features
        return self.fc(context)

# -----------------------------
# Define an Autoencoder for Anomaly Detection
# -----------------------------
class FeatureAutoencoder(nn.Module):
    def __init__(self, feature_dim, latent_dim=64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Number of known classes from training data
num_known_classes = 1 ############ needs fix
input_dim = len(ppg_columns + other_features)  # e.g., 5 features
hidden_dim = 256
num_layers = 3
lr=0.0009226598065126681
model = None 
weight_decay=8.211369649744362e-06
dropout=0.4707771102846937


class PPGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)  # Shape: (n_samples, seq_len, n_features)
        self.y = torch.LongTensor(y)   # Shape: (n_samples,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
# Functions for execution
def train_model():
    global scaler
    global label_encoder
    global model
    global autoencoder
    global weight_decay
    global dropout
    global lr
 

    # Print info
    print('Data processing...')

    # Read newly collected data
    df = pd.read_csv('Logs/' + USER_NAME + '.csv')
    df['User_ID'] = USER_NAME
    # Read mixed unknown users' data
    df1 = pd.read_csv('data/20min_data.csv')
    # df1['User_ID'] = 'unknown'
    # Cancatenate
    dfs = [df1, df]

    data = pd.concat(dfs, ignore_index=True)
    # print(len(data))
    data = data[ppg_columns + other_features + [target]].dropna()

    for col in ppg_columns:
        data[col] = bandpass_filter(data[col])
    
    X_known, y_known = [], []
    X_unknown, y_unknown = [], []

    data[ppg_columns + other_features] = scaler.fit_transform(data[ppg_columns + other_features])
    joblib.dump(scaler, 'scaler_' + USER_NAME + '.save')
    for user_id in data[target].unique():
        user_data = data[data[target] == user_id]
        for i in range(0, len(user_data) - sequence_length, sequence_length // 2):
            seq = user_data.iloc[i:i+sequence_length][ppg_columns + other_features].values
            if user_id in unknown_users:
                X_unknown.append(seq)
                y_unknown.append("unknown")
            else:
                X_known.append(seq)
                # Store as string so that LabelEncoder works with strings
                y_known.append(str(user_id))
    X_known = np.array(X_known)
    y_known = np.array(y_known)
    X_unknown = np.array(X_unknown)


    # Split only the known sequences
    # no need for a test set here
    X_train = np.concatenate([X_known, X_unknown])
    y_train = np.concatenate([y_known, y_unknown])

    y_train_enc = label_encoder.fit_transform(y_train)  # Only known labels
    train_dataset = PPGDataset(X_train, y_train_enc)

    known_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # # Create training dataset (with encoded labels)
    # train_dataset = PPGDataset(X_train, y_train_enc)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create validation split from training data (using 20% of training samples)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train_enc, test_size=0.2, random_state=42)
    train_sub_dataset = PPGDataset(X_train_sub, y_train_sub)
    val_dataset = PPGDataset(X_val, y_val)
    train_sub_loader = DataLoader(train_sub_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)


    # Start training
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')  # change 'cpu' to 'mps' if running on a mac
    num_known_classes = len(label_encoder.classes_)
    model = AttentionBiLSTM(input_dim, hidden_dim, num_layers, num_known_classes, dropout=dropout)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 50

    # Print info
    print('Training started...')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y in train_sub_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.detach().item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
        
        # Calculate training metrics
        train_loss = running_loss / len(train_sub_loader)
        train_acc = 100 * correct_train / total_train
        
        # Validation
        model.eval()
        val_loss, correct_val = 0.0, 0
        total_val = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%')
        print('-' * 50)

    # Save model as model-<USER_NAME> in Logs/
    print('Model being saved as model-' + USER_NAME)
    torch.save(model.state_dict(), 'model-' + USER_NAME + '.pth')
    # Instantiate autoencoder
    feature_dim = hidden_dim * 2  # Dimension of latent features from BiLSTM's context vector
 
    autoencoder = FeatureAutoencoder(feature_dim, latent_dim=64).to(device)
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    # Train autoencoder for a number of epochs (adjust ae_epochs as needed)
    print("Training autoencoder on known latent features:")
    train_autoencoder(model, known_train_loader, ae_optimizer, ae_epochs=epochs, device=device)
    


def evaluate_combined(test_loader, device, threshold=0.5, threshold_ae=0.1):
    global label_encoder
    global model
    if LOAD_MODEL:
        model = AttentionBiLSTM(input_dim, hidden_dim, num_layers, 25, dropout=dropout)
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            max_probs, pred_indices = torch.max(probs, dim=1)
            # Convert numeric predictions to string labels using LabelEncoder
            pred_labels = [label_encoder.inverse_transform([idx.item()])[0] for idx in pred_indices]
            # Apply threshold to determine "unknown"
            final_preds = [pl if conf.item() >= threshold else "unknown"
                           for pl, conf in zip(pred_labels, max_probs)]
            # Append predictions and ground truth (labels stored as strings)
            all_preds.extend(final_preds)
            all_labels.extend(batch_y)

        print("Predictions: ", final_preds)

def inference_model(data, threshold_classifier=0.5, threshold_ae=0.02):
    global scaler
    global label_encoder
    global model
    global autoencoder
    # load model
    # model.load_state_dict(torch.load('Logs/model-' + USER_NAME, weights_only=True))
    # model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    data = data[ppg_columns + other_features].dropna()

    for col in ppg_columns:
        data[col] = bandpass_filter(data[col])
    data[ppg_columns + other_features] = scaler.fit_transform(data[ppg_columns + other_features])
    sequence_array = data[ppg_columns + other_features].values
    sequence = torch.FloatTensor(sequence_array).unsqueeze(0)

    model = model.to(device)

    model.eval()
    threshold = 0.5
    all_preds = []
    all_confidences = []
    

    X_new = data.values.astype(np.float32)
    input_data = torch.tensor(data.values, dtype=torch.float32)
 # shape: [sequence_length, features]
    X_new = np.expand_dims(X_new, axis=0)
# Ensure it's 2D: [batch_size, num_features]
    if input_data.ndim == 1:
        input_data = input_data.unsqueeze(0)

    input_data = input_data.to(device)
    # Get predictions using combined classifier & anomaly detector
    outputs, _ = predict_with_anomaly(model, X_new,
                                    classifier_threshold=threshold_classifier,
                                    ae_threshold=threshold_ae,
                                    device=device)

    # outputs = model(sequence.to(device))  # shape: (batch, num_known_classes)
    probs = torch.softmax(outputs, dim=1)
    max_probs, pred_indices = torch.max(probs, dim=1)
    # Convert numeric predictions to string labels using the LabelEncoder.
    pred_labels = [label_encoder.inverse_transform([idx.item()])[0] for idx in pred_indices]
    # If confidence is below threshold, assign "unknown"
    final_preds = [pl if conf.item() >= threshold else "unknown" 
                    for pl, conf in zip(pred_labels, max_probs)]
    all_preds.extend(outputs)
    all_confidences.extend(max_probs.cpu().detach().numpy())
    print(all_preds, all_confidences)

        


    

    

"""
    Starting interacting with WeBe Band.
"""

async def demo_program():
    """Sends the start command and wait for data transmission.
    """
    global scaler
    queue = asyncio.Queue()
    async def base_data_handler(data):
        global df_live
        # number of entries in the training set
        global train_set_done
        global is_stop
        global ppg_queue
        # EncodedReceived = " ".join(f"0x{n:02x}" for n in data)
        # print(EncodedReceived)

        

        base_index = 6

        assert data[base_index+0] == 0xff
        assert data[base_index+1] == 0xee

        data_dict = {
            # 'timestamp': data[base_index + 2]<<24|data[base_index + 3]<<16|data[base_index + 4]<<8|data[base_index + 5],
            # 'temperature': data[base_index + 6]<<24|data[base_index + 7]<<16|data[base_index + 8]<<8|data[base_index + 9],
            # 'heart_rate_estim': data[base_index + 10]<<8|data[base_index + 11],
            # 'hr_confidence' : data[base_index + 12],
            # 'rr_interbeat_interval': data[base_index + 13]<<8|data[base_index + 14],
            # 'rr_confidence' : data[base_index + 15],
            # 'r_spo2': data[base_index + 16]<<8|data[base_index + 17],
            # 'spo2_confidence': data[base_index + 18],
            # 'spo2_estim': data[base_index + 19]<<8|data[base_index + 20],
            # 'spo2_calc_percentage': data[base_index + 21],
            # 'whrm_suite_curr_opmode': 0,
            # 'spo2_low_sign_quality_flag': data[base_index +22] >> 7 & 0x01,
            # 'spo2_motion_flag':data[base_index +22] >> 6 & 0x01,
            # 'spo2_low_pi_flag': data[base_index +22] >> 5 & 0x01,
            # 'spo2_unreliable_r_flag': data[base_index +22] >> 4 & 0x01,
            # 'spo2_state': data[base_index +22] & 0x03,
            # 'skin_contact_state':data[base_index + 23]&0x02,
            # 'activity_class': data[base_index + 23]>>4,
            # 'walk_steps':data[base_index + 24]<<24|data[base_index + 25]<<16|data[base_index + 26]<<8|data[base_index + 27],
            # 'run_steps': data[base_index + 28]<<24|data[base_index + 29]<<16|data[base_index + 30]<<8|data[base_index + 31],
            # 'kcal': data[base_index + 32]<<24|data[base_index + 33]<<16|data[base_index + 34]<<8|data[base_index + 35],
            # 'cadence': data[base_index + 36]<<24|data[base_index + 37]<<16|data[base_index + 38]<<8|data[base_index + 39],
            # 'event': (data[base_index + 23]>>2) & 0x03,

            'grn_count': data[base_index + 40] << 16 | data[base_index + 41] << 8 | data[base_index + 42],
            'irCnt': data[base_index + 43] << 16 | data[base_index + 44] << 8 | data[base_index + 45],
            'redCnt': data[base_index + 46] << 16 | data[base_index + 47] << 8 | data[base_index + 48],
            'grn2Cnt': data[base_index + 49] << 16 | data[base_index + 50] << 8 | data[base_index + 51],

            # 'accelx': data[base_index + 52] << 8 | data[base_index + 53],
            # 'accely': data[base_index + 54] << 8 | data[base_index + 55],
            # 'accelz': data[base_index + 56] << 8 | data[base_index + 57],

            # 'GSR': data[base_index + 58] << 8 | data[base_index + 59]
        }

        df_temp = pd.DataFrame([data_dict])

        df_live = pd.concat([df_live, df_temp], ignore_index=True)

        base_index = base_index + 60
        for i in range(6):
            data_dict = {
                'grn_count': data[base_index + 0]<<16|data[base_index + 1]<<8|data[base_index + 2],
                'irCnt': data[base_index + 3] << 16 | data[base_index + 4] << 8 | data[base_index + 5],
                'redCnt': data[base_index + 6] << 16 | data[base_index + 7] << 8 | data[base_index + 8],
                'grn2Cnt': data[base_index + 9] << 16 | data[base_index + 10] << 8 | data[base_index + 11],

                # 'accelx': data[base_index + 12] << 8 | data[base_index + 13],
                # 'accely': data[base_index + 14] << 8 | data[base_index + 15],
                # 'accelz': data[base_index + 16] << 8 | data[base_index + 17],

                # 'GSR': data[base_index+18] << 8 | data[base_index+19]
            }

            df_temp = pd.DataFrame([data_dict])
            df_live = pd.concat([df_live, df_temp], ignore_index=True)

            base_index += 20

        if len(df_live) >= MAX_TRAIN_SIZE and not train_set_done and not is_stop and not LOAD_MODEL:
            is_stop = True
            df_live.to_csv(DATA_PATH, index=0)
            train_set_done = True
            await send_command([1, 18]) # stop sending
            await asyncio.sleep(5)
            exit(0)
            train_model()
            
            

            # wait for keyboard input to resume
            print('Model training finished. Press Enter to continue...')
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, sys.stdin.buffer.readline)
            
            # send start command to resume
            await send_command([1, 9, 0, 1, 1]) # resume sending
            await asyncio.sleep(5)
            is_stop = False
            df_live = pd.DataFrame(columns=data_columns)
            
            


        elif train_set_done and not is_stop or LOAD_MODEL:
            if len(df_live) >= sequence_length:
                # inference_model(df_live[:sequence_length]) # construct a frame for authentication
                # if LOAD_MODEL:
                #     scaler = joblib.load("scaler.save")
                
                
                df_slice = df_live[:sequence_length]

                # for col in ppg_columns:
                #     df_slice[col] = bandpass_filter(df_slice[col])

                # 2. Scale the entire DataFrame
                # scaled_array = scaler.transform(df_slice)

                # 3. Wrap scaled data back into a DataFrame (optional, depends on your loader)
                # scaled_df = pd.DataFrame(scaled_array, columns=df_live.columns)

                # 4. Create test loader
                # bandpass and scale done in create_test_loader()
                test_loader = create_test_loader(df_slice)
                                # test_loader = create_test_loader(df_live[:sequence_length])
                device = torch.device('cuda' if torch.cuda.is_available() else 'mps') 
                evaluate_combined(test_loader, device=device, threshold=0.5)
                df_live = pd.DataFrame(columns=data_columns)


    def signal_data_handler(data):
        global df_live

        base_index = 6
        for i in range(9):
            data_dict = {
                'grn_count': data[base_index + 0] << 16 | data[base_index + 1] << 8 | data[base_index + 2],
                'irCnt': data[base_index + 3] << 16 | data[base_index + 4] << 8 | data[base_index + 5],
                'redCnt': data[base_index + 6] << 16 | data[base_index + 7] << 8 | data[base_index + 8],
                'grn2Cnt': data[base_index + 9] << 16 | data[base_index + 10] << 8 | data[base_index + 11],

                # 'accelx': data[base_index + 12] << 8 | data[base_index + 13],
                # 'accely': data[base_index + 14] << 8 | data[base_index + 15],
                # 'accelz': data[base_index + 16] << 8 | data[base_index + 17],

                # 'GSR': data[base_index + 18] << 8 | data[base_index + 19]
            }

            df_temp = pd.DataFrame([data_dict])
            df_live = pd.concat([df_live, df_temp], ignore_index=True)

            base_index = base_index + 20

    def find_uart_device(device: BLEDevice, adv: AdvertisementData):
        # This assumes that the device includes the UART service UUID in the
        # advertising data. This test may need to be adjusted depending on the
        # actual advertising data supplied by the device.
        #print(device)
        if adv.local_name == BLEName:
            print(device)
            queue.put_nowait(device)

    async def send_command(command):
        command = bytearray(command)
        EncodedResponse = " ".join(f"0x{n:02x}" for n in command)
        print("Sent:", EncodedResponse)
        await client.write_gatt_char(UART_RX_CHAR_UUID, command)

    async with BleakScanner(detection_callback=find_uart_device):
        print("Scanning for a device...")
        # this just gets the first device that was queued, then we stop scanning
        device: BLEDevice = await queue.get()


    def handle_disconnect(_: BleakClient):
        global ifLog
        global logFile
        print("Device was disconnected, goodbye.")
        # cancelling all tasks effectively ends the program
        for task in asyncio.all_tasks():
            task.cancel()

    async def handle_rx(_: int, data: bytearray):
        # TODO: update data handling
        global out_counter
        global start_time
        global SentCommand
        global total_bytes
        global df_live
        global time_gap
        total_bytes = total_bytes + len(data)
        
        print("total bytes = %d" % total_bytes)
        if SentCommand or (data[0] == 0x04 and data[1] == 0x01):
            if SentCommand == False:
                start_time = time.time()
                SentCommand = True

        # plot live hr data and do NOT print
        if data[0] == 0x01 and data[1] == 0x90 and len(data) > 42:
            await base_data_handler(data)
        elif data[0] == 0x01 and data[1] == 0x91 and len(data) > 42:
            signal_data_handler(data)

        elif data[0] == 0x01 and data[1] == 0x99 and len(data)== 6:
            df_live.to_csv("Logs/" + USER_NAME + ".csv", index=0)
        elif data[0] == 0x05 and data[1] == 0x02 and len(data) >=6:
            wearable_time = data[4]<<24|data[5]<<16|data[6]<<8|data[7]
            time_gap = int(time.time())-wearable_time
        else:
            DecodedResponse = " ".join(f"0x{n:02x}" for n in data)
            print("received:", out_counter, " >> ", DecodedResponse)
        out_counter = out_counter + 1

    async with BleakClient(device, disconnected_callback=handle_disconnect) as client:
        await client.start_notify(UART_TX_CHAR_UUID, handle_rx)
        

        await send_command([5, 2, 0, 0])  # get time

        # set time
        time_command = bytearray([5, 1, 0, 4])
        cur_time = time.time()
        for i in range(3, -1, -1):
            time_command.append(int(cur_time)>>(8*i) & 0xff)
        EncodedResponse = " ".join(f"0x{n:02x}" for n in time_command)
        print("Sent:", EncodedResponse)
        await client.write_gatt_char(UART_RX_CHAR_UUID, time_command)

        await asyncio.sleep(5) #  must add

        await send_command([1, 9, 0, 1, 1]) # start measuring

        await asyncio.Future()  # receive data forever






# It is important to use asyncio.run() to get proper cleanup on KeyboardInterrupt.
# This was introduced in Python 3.7. If you need it in Python 3.6, you can copy
# it from https://github.com/python/cpython/blob/3.7/Lib/asyncio/runners.py

if __name__ == "__main__":
    out_counter = 0

    try:
        asyncio.run(demo_program())
       
    except asyncio.CancelledError:
        # task is cancelled on disconnect, so we ignore this error
        pass
