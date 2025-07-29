# 🚀 Asset Time Series Prediction Platform

A sophisticated machine learning platform for asset (e.g., cryptocurrency or financial) price movement prediction using a **Transformer** architecture, with automated model retraining and real-time inference capabilities. All sensitive business logic, feature engineering, and asset mapping are anonymized for public portfolio use.

## 🌟 Features

- **Advanced ML Architecture**: Implements transformer-based model for time series forecasting (anonymized feature set)
- **Automated Pipeline**: Complete MLOps pipeline with training, inference, and retraining automation
- **Real-time Predictions**: Flask API for real-time asset trend predictions
- **Cloud Integration**: Seamless integration with Google Cloud Storage and Supabase
- **Model Monitoring**: Automated accuracy monitoring with threshold-based retraining
- **Scalable Design**: Configurable and anonymized feature engineering and model parameters

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Flask API     │────│  ML Pipelines    │────│  Data Storage   │
│  - /timeseries  │    │  - Training      │    │  - Database     │
│  - /verify      │    │  - Inference     │    │  - Cloud Storage│
│  - /retrain     │    │  - Retraining    │    │  - Local Models │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **ML Framework**: PyTorch, scikit-learn, Transformers
- **Database**: Supabase (PostgreSQL)
- **Cloud Storage**: Google Cloud Storage
- **Hyperparameter Tuning**: Optuna
- **Data Processing**: Pandas, NumPy
- **Deployment**: Docker, Gunicorn

## 📦 Project Structure

```
├── app.py                          # Flask API entry point
├── config.py                       # Environment configuration (anonymized)
├── requirements.txt                # Dependencies
├── Dockerfile                      # Container configuration
│
├── pipelines/                      # ML pipelines
│   ├── pipeline_timeseries.py      # Time series prediction pipeline
│   ├── pipeline_verify.py          # Prediction verification pipeline
│   ├── pipeline_retrain.py         # Automated retraining pipeline
│   ├── pipeline_incremental.py     # Incremental learning pipeline
│   ├── pipeline_labeling.py        # Data labeling pipeline
│   └── __init__.py
│
├── models/                         # Model definitions and storage
│   ├── timeseries_model.py         # Transformer model architecture (anonymized)
│   ├── *.pt                        # Trained model weights
│   └── *.pkl                       # Feature scalers
│
├── data/                           # Data processing modules
│   ├── preprocess.py               # Data preprocessing utilities (anonymized features)
│   └── supabase_io.py              # Database I/O operations
│
├── inference/                      # Model inference
│   └── timeseries_inference.py     # Prediction execution
│
├── trainers/                       # Model training
│   └── train_patchtst.py           # Model training script
│
├── utils/                          # Utility modules
│   ├── gcs_utils.py                # Google Cloud Storage utilities
│   └── training_utils.py           # Training helper functions (anonymized hyperparameters)
│
└── debug_tools/                    # Debugging/diagnostics tools
    └── debug_tools.py
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Database system (Supabase recommended)
- Cloud storage bucket (for production)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file with your configurations
   ENV_TYPE=local
   SUPABASE_URL=your_database_url
   SUPABASE_KEY=your_database_key
   # ...other required variables (see config.py)
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## 📡 API Endpoints

### `POST /timeseries`
Triggers the time series prediction pipeline
```bash
curl -X POST http://localhost:8080/timeseries
```

### `POST /verify`
Runs prediction verification against historical data
```bash
curl -X POST http://localhost:8080/verify
```

### `POST /retrain`
Initiates model retraining based on performance metrics
```bash
curl -X POST http://localhost:8080/retrain
```

### `POST /incremental`
Triggers incremental learning pipeline
```bash
curl -X POST http://localhost:8080/incremental
```

### `POST /labeling`
Triggers data labeling pipeline
```bash
curl -X POST http://localhost:8080/labeling
```

### `GET /`
Health check endpoint
```bash
curl http://localhost:8080/
```

## 🧠 Model & Feature Details (Anonymized)

### Transformer Architecture
- **Attention-based approach**: Leverages self-attention mechanisms for temporal dependencies
- **Configurable & anonymized**: Model dimensions, layers, and hyperparameters are anonymized for public release
- **Multi-feature input**: Processes multiple anonymized features simultaneously
- **Sequence modeling**: Handles variable-length time series data

### Supported Assets
- Generic asset mapping system (anonymized)
- Easily extensible to additional financial instruments

### Feature Engineering
- Multiple anonymized features (see `config.py`)
- Automated feature scaling and normalization
- All feature names and logic are anonymized for portfolio use

## 🔄 MLOps Pipeline

1. **Data Ingestion**: Automated fetching of market data and indicators
2. **Model Training**: Transformer training with anonymized hyperparameter optimization
3. **Model Deployment**: Automatic model versioning and cloud storage
4. **Inference**: Real-time prediction generation
5. **Monitoring**: Performance tracking and automated retraining triggers
6. **Verification**: Historical prediction validation

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t asset-prediction-platform .

# Run the container
docker run -p 8080:8080 --env-file .env asset-prediction-platform
```

## 📈 Performance Monitoring

The platform includes automated performance monitoring:
- **Configurable thresholds**: Adjustable accuracy requirements (anonymized in config)
- **Rolling evaluation**: Time-window based performance calculation
- **Auto-retraining**: Triggered when performance drops below thresholds
- **Model versioning**: Timestamped model artifacts in cloud storage

## ⚙️ Configuration

Key configuration options (all anonymized):
- Model architecture parameters
- Training hyperparameters
- Performance thresholds
- Feature selection
- Data processing parameters

## 🔒 Security & Privacy

- Environment-based configuration management
- Secure credential handling
- All business logic, features, and parameters are anonymized for public release

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

*A sophisticated machine learning platform demonstrating advanced time series forecasting with transformer architecture and MLOps best practices. All sensitive business logic and features are anonymized for portfolio sharing.*