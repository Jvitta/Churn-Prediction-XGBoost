import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint

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
    
    df_selected.to_csv('cleaned_churn_dataset.csv', index=False)
    
    return df_selected

def train_model_with_informed_search():
    """
    Train and evaluate the XGBoost model with a two-stage informed search
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
    # Save training and test sets to CSV files
    print("Saving train and test sets to CSV...")
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('train_set.csv', index=False)
    test_data.to_csv('test_set.csv', index=False)
    
    # STAGE 1: Broad search with more iterations
    print("\n=== STAGE 1: Broad Parameter Search ===")
    
    # Define the broad parameter space with more options
    broad_param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(2, 12),
        'learning_rate': uniform(0.005, 0.3),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 1.0),
        'scale_pos_weight': [1, 2, 3, 5],
    }
    
    # Initialize the model
    base_model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # Initialize RandomizedSearchCV for broad search
    print("Starting broad RandomizedSearchCV...")
    broad_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=broad_param_dist,
        n_iter=300,
        scoring=['f1', 'roc_auc', 'precision', 'recall'],
        refit='f1',
        n_jobs=-1,
        cv=5,
        verbose=0,  # Changed from 2 to 0 to remove CV output
        random_state=42,
        return_train_score=True
    )
    
    # Fit broad search
    broad_search.fit(X_train, y_train)
    
    # Get the best parameters from broad search
    best_broad_params = broad_search.best_params_
    print("\nBest parameters from broad search:")
    for param, value in best_broad_params.items():
        print(f"{param}: {value}")
    
    # STAGE 2: Refined search around best parameters
    print("\n=== STAGE 2: Refined Parameter Search ===")
    
    # Define refined parameter space based on best parameters from broad search
    refined_param_dist = {
        'n_estimators': randint(
            max(50, best_broad_params['n_estimators'] - 100),
            best_broad_params['n_estimators'] + 100
        ),
        'max_depth': randint(
            max(1, best_broad_params['max_depth'] - 2),
            best_broad_params['max_depth'] + 2
        ),
        'learning_rate': uniform(
            max(0.001, best_broad_params['learning_rate'] - 0.05),
            min(0.1, best_broad_params['learning_rate'] + 0.05)
        ),
        'subsample': uniform(
            max(0.5, best_broad_params['subsample'] - 0.1),
            min(0.3, 1.0 - best_broad_params['subsample'])
        ),
        'colsample_bytree': uniform(
            max(0.5, best_broad_params['colsample_bytree'] - 0.1),
            min(0.3, 1.0 - best_broad_params['colsample_bytree'])
        ),
        'min_child_weight': randint(
            max(1, best_broad_params['min_child_weight'] - 2),
            best_broad_params['min_child_weight'] + 3
        ),
        'gamma': uniform(
            max(0, best_broad_params['gamma'] - 0.2),
            min(0.4, best_broad_params['gamma'] + 0.4)
        ),
        'reg_alpha': uniform(0, 1.0),
        'reg_lambda': uniform(0.5, 5.0),
        'scale_pos_weight': [best_broad_params.get('scale_pos_weight', 1), 
                            best_broad_params.get('scale_pos_weight', 1) + 1,
                            best_broad_params.get('scale_pos_weight', 1) - 1],
    }
    
    # Initialize RandomizedSearchCV for refined search
    print("Starting refined RandomizedSearchCV...")
    refined_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=refined_param_dist,
        n_iter=300,
        scoring=['f1', 'roc_auc', 'precision', 'recall'],
        refit='f1',
        n_jobs=-1,
        cv=5,
        verbose=0,  
        random_state=42,
        return_train_score=True
    )
    
    # Fit refined search
    refined_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = refined_search.best_estimator_
    
    # Print best parameters
    print("\nBest parameters from refined search:")
    for param, value in refined_search.best_params_.items():
        print(f"{param}: {value}")
    
    # Make predictions with best model
    print("\nMaking predictions with best model...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
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
    plt.title('Confusion Matrix (Best Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_informed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance (Best Model)')
    plt.tight_layout()
    plt.savefig('feature_importance_informed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot search results
    plt.figure(figsize=(12, 8))
    results = pd.DataFrame(refined_search.cv_results_)
    plt.scatter(results['mean_test_f1'], results['mean_test_roc_auc'], c=results['rank_test_f1'], cmap='viridis')
    plt.colorbar(label='Rank')
    plt.xlabel('F1 Score')
    plt.ylabel('ROC AUC')
    plt.title('Hyperparameter Search Results')
    plt.tight_layout()
    plt.savefig('search_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the model
    best_model.save_model('best_xgboost_model.json')
    
    # Save search results
    results = pd.DataFrame(refined_search.cv_results_)
    results.to_csv('hyperparameter_search_results.csv', index=False)
    
    return best_model, importance_df, refined_search.best_params_

if __name__ == "__main__":
    best_model, importance_df, best_params = train_model_with_informed_search()
    print("\nFeature Importance:")
    print(importance_df)
    print("\nModel training complete. Visualizations saved as PNG files")
    print("Best model saved as 'best_xgboost_model.json'")
