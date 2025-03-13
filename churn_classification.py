import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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

def train_model():
    """
    Train and evaluate the XGBoost model
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
    
    # Initialize and train model
    print("Training XGBoost model...")
    model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, importance_df

if __name__ == "__main__":
    model, importance_df = train_model()
    print("\nFeature Importance:")
    print(importance_df)
    print("\nModel training complete. Visualizations saved as 'confusion_matrix.png' and 'feature_importance.png'")
