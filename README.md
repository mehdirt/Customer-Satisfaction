# Customer Satisfaction Prediction System

A comprehensive machine learning pipeline for predicting customer satisfaction scores based on order and product data. This project implements a complete MLOps workflow using ZenML for pipeline management, MLflow for experiment tracking, and Streamlit for interactive model deployment.

## 🎯 Project Overview

This project focuses on predicting customer satisfaction scores (ranging from 0-5) based on various order and product features. The system uses a Linear Regression model trained on historical customer order data, with features including payment information, product details, and shipping characteristics.

## 🚀 Features

### Data Processing Pipeline
- **Automated Data Ingestion and Cleaning**
  - Removes unnecessary temporal features (order timestamps)
  - Handles missing values using median imputation
  - Processes text features (review comments)
  - Filters numeric features and removes irrelevant columns

- **Data Splitting Strategy**
  - Implements train-test split with 80-20 ratio
  - Maintains consistent random state for reproducibility
  - Separates features and target variable (review_score)

### Machine Learning Pipeline
- **Linear Regression Implementation**
  - Custom model class with abstract base class
  - Configurable hyperparameters
  - Comprehensive error handling and logging

- **Model Evaluation System**
  - Multiple evaluation metrics:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R-squared (R2) Score
  - Detailed logging of evaluation results
  - Abstract evaluation strategy for extensibility

### MLOps Integration
- **ZenML Pipeline Management**
  - Structured pipeline steps for data processing
  - Model training and evaluation workflows
  - Deployment pipeline configuration

- **MLflow Integration**
  - Experiment tracking
  - Model versioning
  - Performance metric logging

### Interactive Web Interface
- **Streamlit Application**
  - Real-time prediction interface
  - Input features:
    - Payment details (sequential, installments, value)
    - Product characteristics (price, dimensions, weight)
    - Product information (name length, description length, photos)
  - Visual pipeline explanation
  - Feature descriptions and documentation

## 🛠️ Technologies Used

### Core ML Stack
- **scikit-learn**: 
  - LinearRegression for model implementation
  - train_test_split for data splitting
  - Evaluation metrics (MSE, RMSE, R2)
- **pandas**: Data manipulation and preprocessing
- **numpy**: Numerical computations and array operations

### MLOps Tools
- **ZenML**: 
  - Pipeline orchestration
  - Step management
  - Model deployment
- **MLflow**: 
  - Model tracking
  - Experiment management
  - Model serving
- **Docker**: Containerization for deployment

### Web Interface
- **Streamlit**: 
  - Interactive web application
  - Real-time prediction interface
  - Data visualization
- **Pillow**: Image processing for pipeline visualization

### Development Tools
- **Pydantic**: Data validation and configuration
- **Click**: Command-line interface for deployment
- **Rich**: Terminal formatting and logging

## 📁 Project Structure

```
.
├── data/                  # Raw and processed data
├── src/                   # Core source code
│   ├── data_cleaning.py   # Data preprocessing strategies
│   │   ├── DataStrategy (ABC)
│   │   ├── DataPreProcessStrategy
│   │   └── DataSplitStrategy
│   ├── evaluation.py      # Model evaluation metrics
│   │   ├── Evalueation (ABC)
│   │   ├── MSE
│   │   ├── RMSE
│   │   └── R2
│   └── model_dev.py       # Model development
│       ├── Model (ABC)
│       └── LinearRegressionModel
├── steps/                 # Pipeline steps
│   ├── clean_data.py      # Data cleaning step
│   ├── evaluation.py      # Model evaluation step
│   ├── ingest_data.py     # Data ingestion step
│   ├── model_train.py     # Model training step
│   └── config.py          # Configuration management
├── pipelines/             # Pipeline definitions
│   ├── training_pipeline.py
│   ├── deployment_pipeline.py
│   └── utils.py
├── saved_model/           # Trained model artifacts
├── assets/                # Static assets for web interface
├── streamlit_app.py       # Web application
├── run_pipeline.py        # Pipeline execution script
└── run_deployment.py      # Deployment management
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- Docker (for deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-satisfaction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Pipeline

1. Initialize ZenML:
```bash
zenml init
```

2. Run the training pipeline:
```bash
python run_pipeline.py
```

3. Deploy the model:
```bash
python run_deployment.py
```

4. Start the web interface:
```bash
streamlit run streamlit_app.py
```

## 📊 Model Features and Predictions

The model uses the following features for prediction:

| Feature | Description | Type |
|---------|-------------|------|
| Payment Sequential | Sequence number for multiple payment methods | Integer |
| Payment Installments | Number of payment installments | Integer |
| Payment Value | Total payment amount | Float |
| Price | Product price | Float |
| Freight Value | Shipping cost | Float |
| Product Name Length | Length of product name | Integer |
| Product Description Length | Length of product description | Integer |
| Product Photos Quantity | Number of product photos | Integer |
| Product Weight | Weight in grams | Float |
| Product Length | Length in centimeters | Float |
| Product Height | Height in centimeters | Float |
| Product Width | Width in centimeters | Float |

The model predicts a customer satisfaction score on a scale of 0-5.


## Project Demo

- Explore the web app to see the project in action: **[Live Demo](https://customer-satisfaction-7gsdkyqbbwhrkaydmeudez.streamlit.app/)**


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contact Information
If you have any feedback, feel free to reach out.

[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mahdirafati680@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahdi-rafati-97420a197/)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@mehdirt)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/itsmehdirt)
