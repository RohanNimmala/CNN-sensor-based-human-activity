import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import json

# Define the model architecture
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class HybridCNNWithAttention(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(HybridCNNWithAttention, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.attention = ChannelAttention(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.attention(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# Constants
MODEL_PATH = 'har_model.pth'
SCALER_PATH = 'har_scaler.pkl'
NUM_CHANNELS = 9
NUM_CLASSES = 6
ACTIVITY_LABELS = {
    0: 'Walking',
    1: 'Walking Upstairs',
    2: 'Walking Downstairs',
    3: 'Sitting',
    4: 'Standing',
    5: 'Laying'
}

# Initialize Flask application
app = Flask(__name__)

# Configure for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'npy'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global model and device variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
scaler = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler
    
    model = HybridCNNWithAttention(num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Model file {MODEL_PATH} not found, model will be trained from scratch")
    
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded from {SCALER_PATH}")
        
def train_model(X_path, y_path):
    """Train the model on new data"""
    global model, scaler
    
    try:
        X = np.load(X_path)
        y = np.load(y_path)
        
        # Initialize model if not already done
        if model is None:
            model = HybridCNNWithAttention(num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES).to(device)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(10):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())
        
        report = classification_report(all_labels, all_preds, zero_division=0)
        print("\nClassification Report:\n", report)
        
        # Save the model
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        
        return {"status": "success", "message": "Model trained successfully", "report": report}
    
    except Exception as e:
        return {"status": "error", "message": f"Error training model: {str(e)}"}

def predict(data):
    """Make predictions using the trained model"""
    global model
    
    if model is None:
        return {"status": "error", "message": "Model not loaded. Please train the model first."}
    
    try:
        # Ensure data is in the correct format for the model
        if isinstance(data, list):
            data = np.array(data)
        
        if len(data.shape) == 2:  # Single sample
            data = np.expand_dims(data, axis=0)
        
        # Convert to tensor and make prediction
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(data_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        results = []
        for i, pred in enumerate(preds):
            results.append({
                "prediction": int(pred),
                "activity": ACTIVITY_LABELS.get(int(pred), f"Unknown Activity {pred}"),
                "confidence": float(probs[i][pred])
            })
        
        return {"status": "success", "predictions": results}
    
    except Exception as e:
        return {"status": "error", "message": f"Error making prediction: {str(e)}"}

# API endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if the server is running"""
    return jsonify({"status": "online", "model_loaded": model is not None})

@app.route('/api/train', methods=['POST'])
def api_train():
    """Train the model using data files"""
    if 'X_file' not in request.files or 'y_file' not in request.files:
        return jsonify({"status": "error", "message": "X_file and y_file are required"})
    
    X_file = request.files['X_file']
    y_file = request.files['y_file']
    
    if X_file.filename == '' or y_file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})
    
    if not (allowed_file(X_file.filename) and allowed_file(y_file.filename)):
        return jsonify({"status": "error", "message": "File type not allowed"})
    
    X_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(X_file.filename))
    y_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(y_file.filename))
    
    X_file.save(X_path)
    y_file.save(y_path)
    
    result = train_model(X_path, y_path)
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Make predictions on new data"""
    if request.is_json:
        # JSON input format
        content = request.get_json()
        if 'data' not in content:
            return jsonify({"status": "error", "message": "No data field in JSON"})
        
        data = content['data']
        result = predict(data)
        return jsonify(result)
    elif 'file' in request.files:
        # File upload format
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"})
        
        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "File type not allowed"})
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        
        try:
            data = np.load(file_path)
            result = predict(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error processing file: {str(e)}"})
    else:
        return jsonify({"status": "error", "message": "No data provided"})

if __name__ == "__main__":
    # Load the model if available
    load_model()
    
    # If model is not loaded, try to train with default files if they exist
    if model is None and os.path.exists("synthetic_har_data_X.npy") and os.path.exists("synthetic_har_data_y.npy"):
        print("Training model with default data files...")
        train_model("synthetic_har_data_X.npy", "synthetic_har_data_y.npy")
    
    # Run the Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)