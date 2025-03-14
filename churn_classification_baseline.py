import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prepare_data(filepath='churn_clean.csv'):
    """
    Load the data and select only the chosen features
    """
    # Selected features from RFECV
    selected_features = [
        'Tenure', 'MonthlyCharge', 'StreamingMovies', 'StreamingTV',
        'Contract_Two Year', 'Contract_One year', 'Multiple',
        'InternetService_Fiber Optic', 'Item4', 'Phone',
        'PaymentMethod_Credit Card (automatic)', 'Churn'
    ]
    
    # Load preprocessed data
    df = pd.read_csv(filepath)
    
    # Convert boolean columns to int
    bool_columns = ['StreamingMovies', 'StreamingTV', 'Multiple', 'Phone', 'Churn']
    for col in bool_columns:
        df[col] = (df[col] == 'Yes').astype(int)
    
    # Create necessary dummy variables
    df = pd.get_dummies(df, columns=['Contract', 'InternetService', 'PaymentMethod'])
    
    # Select only the features we want
    df_selected = df[selected_features]
    
    return df_selected

def train_baseline_model():
    """
    Train and evaluate the XGBoost model with default parameters
    """
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split into train and test sets
    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize the model with default parameters
    print("\nTraining XGBoost model with default parameters...")
    model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Calculate PR AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"PR AUC Score: {pr_auc:.4f}")
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Baseline Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance (Baseline Model)')
    plt.tight_layout()
    plt.savefig('feature_importance_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the model
    model.save_model('baseline_xgboost_model.json')
    
    # Print default parameters
    print("\nDefault XGBoost Parameters:")
    default_params = model.get_params()
    important_params = {
        'n_estimators': default_params['n_estimators'],
        'max_depth': default_params['max_depth'],
        'learning_rate': default_params['learning_rate'],
        'subsample': default_params['subsample'],
        'colsample_bytree': default_params['colsample_bytree'],
        'min_child_weight': default_params['min_child_weight'],
        'gamma': default_params['gamma'],
        'reg_alpha': default_params['reg_alpha'],
        'reg_lambda': default_params['reg_lambda'],
        'scale_pos_weight': default_params['scale_pos_weight']
    }
    for param, value in important_params.items():
        print(f"{param}: {value}")
    
    return model, importance_df, important_params

if __name__ == "__main__":
    model, importance_df, default_params = train_baseline_model()
    print("\nFeature Importance:")
    print(importance_df)
    print("\nModel training complete. Visualizations saved as PNG files")
    print("Baseline model saved as 'baseline_xgboost_model.json'")
