# ML Pipeline

A comprehensive machine learning pipeline for data processing, model training, evaluation, and deployment.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Pipeline Stages](#pipeline-stages)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Contributing](#contributing)
- [License](#license)

## Overview

This ML pipeline provides an end-to-end solution for machine learning workflows, from data ingestion and preprocessing to model training, evaluation, and deployment. It's designed to be scalable, reproducible, and easy to maintain.

## Features

- **Automated Data Pipeline**: Streamlined data ingestion, cleaning, and preprocessing
- **Model Training**: Support for multiple ML algorithms and hyperparameter tuning
- **Experiment Tracking**: Built-in logging and experiment management
- **Model Evaluation**: Comprehensive metrics and validation procedures
- **Deployment Ready**: Easy deployment to various platforms
- **Monitoring & Logging**: Real-time performance monitoring and logging
- **Scalable Architecture**: Designed to handle large datasets and complex models

## Project Structure

```
ml_pipeline/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── config.yaml
│   ├── model_config.yaml
│   └── data_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   └── feature_selection.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   ├── predict_model.py
│   │   └── model_utils.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── helpers.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_utils.py
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_experiments.ipynb
│   └── results_analysis.ipynb
├── models/
│   ├── trained_models/
│   └── model_artifacts/
├── logs/
│ 
└── scripts/
    ├── run_pipeline.py
    ├── train.py
    └── deploy.py
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda
- Git

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/sobhan2204/ml_pipeline.git
cd ml_pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Basic Pipeline Execution

Run the complete pipeline:
```bash
python scripts/run_pipeline.py --config config/config.yaml
```

### Individual Components

**Data Processing:**
```bash
python -m src.data.preprocessing --input data/raw/dataset.csv --output data/processed/
```

**Model Training:**
```bash
python -m src.models.train_model --config config/model_config.yaml --data data/processed/
```

**Model Prediction:**
```bash
python -m src.models.predict_model --model models/trained_models/best_model.pkl --input data/test.csv
```

### Python API Usage

```python
from src.models.train_model import MLPipeline
from src.data.data_loader import DataLoader

# Initialize pipeline
pipeline = MLPipeline(config_path='config/config.yaml')

# Load and process data
data_loader = DataLoader()
train_data, test_data = data_loader.load_data('data/raw/dataset.csv')

# Train model
pipeline.train(train_data)

# Make predictions
predictions = pipeline.predict(test_data)
```

## Configuration

The pipeline uses YAML configuration files for easy customization:

### Main Configuration (`config/config.yaml`)
```yaml
data:
  raw_data_path: "data/raw/"
  processed_data_path: "data/processed/"
  test_size: 0.2
  validation_size: 0.1
  
preprocessing:
  scaling: "StandardScaler"
  encoding: "OneHotEncoder"
  handle_missing: "median"
  
model:
  algorithm: "RandomForestClassifier"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42
    
training:
  cross_validation: 5
  scoring: "f1_weighted"
  
logging:
  level: "INFO"
  log_file: "logs/pipeline.log"
```

## Pipeline Stages

### 1. Data Ingestion
- Load data from various sources (CSV, databases, APIs)
- Data quality checks and validation
- Data versioning and lineage tracking

### 2. Data Preprocessing
- Missing value handling
- Feature scaling and normalization
- Categorical encoding
- Data splitting (train/validation/test)

### 3. Feature Engineering
- Feature creation and transformation
- Feature selection using statistical methods
- Dimensionality reduction (PCA, t-SNE)

### 4. Model Training
- Multiple algorithm support (scikit-learn, XGBoost, etc.)
- Hyperparameter optimization (Grid Search, Random Search, Bayesian)
- Cross-validation and model selection

### 5. Model Evaluation
- Performance metrics calculation
- Model comparison and validation
- Visualization of results

### 6. Model Deployment
- Model serialization and versioning
- API endpoint creation
- Containerization support

## Model Training

### Supported Algorithms

- **Classification**: Logistic Regression, Random Forest, SVM, XGBoost, Neural Networks
- **Regression**: Linear Regression, Ridge, Lasso, Random Forest Regressor, XGBoost
- **Clustering**: K-Means, DBSCAN, Hierarchical Clustering

### Hyperparameter Tuning

```python
from src.models.train_model import HyperparameterTuner

tuner = HyperparameterTuner(
    model_type='RandomForestClassifier',
    search_type='bayesian',
    cv_folds=5
)

best_params = tuner.tune(X_train, y_train)
```

## Evaluation

### Metrics Supported

- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- **Regression**: MSE, RMSE, MAE, R², Residual Analysis
- **Clustering**: Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index

### Model Comparison

```python
from src.evaluation.metrics import ModelComparator

comparator = ModelComparator()
results = comparator.compare_models(models_dict, X_test, y_test)
comparator.plot_comparison(results)
```

## Deployment

### Local Deployment

```bash
python scripts/deploy.py --model models/trained_models/best_model.pkl --port 8000
```
### Logging

All pipeline activities are logged with different levels:
- INFO: General pipeline progress
- DEBUG: Detailed debugging information
- ERROR: Error messages and stack traces
- WARNING: Performance warnings and alerts

## API Documentation

### REST API Endpoints

- `POST /predict`: Make predictions on new data
- `GET /health`: Health check endpoint
- `GET /metrics`: Model performance metrics
- `POST /retrain`: Trigger model retraining

### Example API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

## Testing

Run the test suite:
```bash
pytest tests/ -v --coverage
```

Run specific test categories:
```bash
# Test data processing
pytest tests/test_data.py

# Test model training
pytest tests/test_models.py

# Test utilities
pytest tests/test_utils.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Troubleshooting

### Common Issues

**Issue**: ModuleNotFoundError when running the pipeline
**Solution**: Ensure you've installed the package with `pip install -e .`

**Issue**: CUDA out of memory errors
**Solution**: Reduce batch size in the configuration or use CPU training

**Issue**: Poor model performance
**Solution**: Check data quality, try feature engineering, or tune hyperparameters

## Changelog

### v1.0.0 (Latest)
- Initial release
- Core pipeline functionality
- Model training and evaluation
- Basic deployment support

## Acknowledgments

- Thanks to the open-source community for the amazing ML libraries
- Special thanks to contributors and maintainers
- Inspired by MLOps best practices and industry standards

## Contact

- **Author**: Sobhan
- **Email**: [pandasobhan22@gmail.com]
- **GitHub**: [@sobhan2204](https://github.com/sobhan2204)
- **Issues**: [GitHub Issues](https://github.com/sobhan2204/ml_pipeline/issues)

---

**Note**: This README assumes a standard ML pipeline structure. Please modify the sections according to your specific implementation, dependencies, and use case.
