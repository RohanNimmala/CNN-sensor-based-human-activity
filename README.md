# Human Activity Recognition (HAR) Server

This project implements a server for Human Activity Recognition using sensor data. The system uses a Hybrid CNN with Channel Attention mechanism to classify different activities based on time-series sensor data.

## Project Structure

- `har_server.py`: The main Flask server that handles training and prediction requests
- `convert_csv.py`: Utility script to convert CSV data to the NumPy format required by the model
- `requirements.txt`: List of required Python packages
- `run_server.py`: Original training script (for reference)

## Setup Instructions

### 1. Create a Virtual Environment

```bash
python3 -m venv har_env
source har_env/bin/activate  # On Windows: har_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Convert Data (if starting with CSV)

If your data is in CSV format, you can convert it to the NumPy format required by the model:

```bash
python convert_csv.py synthetic_har_data.csv
```

This will create:

- `synthetic_har_data_X.npy`: Feature data
- `synthetic_har_data_y.npy`: Label data
- `har_scaler.pkl`: Data scaler (optional)

### 4. Run the Server

```bash
python har_server.py
```

The server will:

1. Look for a pre-trained model (`har_model.pth`)
2. If no model is found, it will attempt to train using the default data files
3. Start on port 5000 (http://localhost:5000)

## API Endpoints

### Health Check

```
GET /api/health
```

Response:

```json
{
  "status": "online",
  "model_loaded": true
}
```

### Train Model

```
POST /api/train
```

Form data:

- `X_file`: NumPy file containing feature data
- `y_file`: NumPy file containing label data

Response:

```json
{
  "status": "success",
  "message": "Model trained successfully",
  "report": "Classification report details..."
}
```

### Make Predictions

```
POST /api/predict
```

Option 1: JSON payload:

```json
{
  "data": [...]  # Array of sensor readings
}
```

Option 2: File upload:

- `file`: NumPy file containing data to predict

Response:

```json
{
  "status": "success",
  "predictions": [
    {
      "prediction": 0,
      "activity": "Walking",
      "confidence": 0.92
    },
    ...
  ]
}
```

## Activity Labels

The model classifies activities into these categories:

- 0: Walking
- 1: Walking Upstairs
- 2: Walking Downstairs
- 3: Sitting
- 4: Standing
- 5: Laying

## Input Data Format

The model expects sensor data in the format:

- Shape: (samples, channels, timesteps)
- Channels: 9 (typically 3-axis accelerometer, gyroscope, and magnetometer)
- Each sample represents a window of sensor readings

## Example Usage

### Training with new data:

```bash
curl -X POST -F "X_file=@new_data_X.npy" -F "y_file=@new_data_y.npy" http://localhost:5000/api/train
```

### Making predictions with a JSON payload:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"data": [...]}' http://localhost:5000/api/predict
```

### Making predictions with a file:

```bash
curl -X POST -F "file=@test_data.npy" http://localhost:5000/api/predict
```
