# Customer Churn Analysis Project

## Overview
This project implements a machine learning solution for predicting customer churn using XGBoost with different hyperparameter optimization approaches. The project includes data preprocessing, exploratory data analysis (EDA), and two model implementations:
- Baseline model with default parameters
- Informed search using RandomizedSearchCV

## Project Structure
```python
.
├── README.md
├── churn_preprocessing.py    # Data preprocessing and feature selection
├── churn_classification.py   # Informed search implementation
├── churn_classification_baseline.py  # Baseline model
├── churn_classification_genetic.py   # Genetic algorithm implementation
├── eda.py                   # Exploratory data analysis
├── environment.yml          # Conda environment specification
└── churn_clean.csv          # Input dataset
```

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-churn-analysis
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate churn_analysis
```

## Usage

### 1. Data Preprocessing
Run the preprocessing script to clean the data and perform feature selection:
```bash
python churn_preprocessing.py
```
This will:
- Clean the raw data
- Encode categorical features
- Perform feature selection using mutual information and RFECV
- Generate feature importance visualizations

### 2. Exploratory Data Analysis
Run the EDA script to generate correlation analysis:
```bash
python eda.py
```
This will create a correlation heatmap and identify the most important features for churn prediction.

### 3. Model Training and Evaluation

#### Baseline Model
Run the baseline model with default parameters:
```bash
python churn_classification_baseline.py
```

#### Informed Search Model
Run the informed search implementation:
```bash
python churn_classification.py
```

#### Genetic Algorithm Model
Run the genetic algorithm implementation:
```bash
python churn_classification_genetic.py
```

## Model Implementations

### 1. Baseline Model
- Uses XGBoost with default parameters
- Provides a performance benchmark
- Generates basic evaluation metrics and visualizations

### 2. Informed Search Model
- Implements a two-stage hyperparameter optimization
- Stage 1: Broad parameter search
- Stage 2: Refined search around promising parameters
- Uses RandomizedSearchCV with multiple scoring metrics

### 3. Genetic Algorithm Model
- Uses evolutionary algorithms for hyperparameter optimization
- Implements natural selection-inspired parameter tuning
- Provides visualization of the evolution process

## Output Files

Each model implementation generates:
- Confusion matrix plot
- Feature importance visualization
- Model performance metrics
- Trained model file (JSON format)
- Detailed results CSV

## Dependencies
- Python 3.11
- pandas
- numpy
- scikit-learn
- xgboost
- seaborn
- matplotlib
- deap (for genetic algorithm)

## Results
The project generates various visualization files:
- `confusion_matrix_*.png`: Confusion matrices for each model
- `feature_importance_*.png`: Feature importance plots
- `mutual_information.png`: Mutual information scores
- `correlation_heatmap_with_categories.png`: Feature correlation analysis
- `genetic_evolution.png`: Evolution of genetic algorithm performance

## Model Comparison
Each implementation provides:
- Classification report
- ROC AUC score
- PR AUC score
- Feature importance rankings

## Contributing
This project is part of a WGU course assignment. While it's not open for direct contributions, feel free to fork the repository and adapt it for your own use.

## License
This project is created for educational purposes as part of WGU coursework.

## Author
Created by [Your Name] for WGU D603 course.
