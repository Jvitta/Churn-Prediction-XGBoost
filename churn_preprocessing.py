import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, f_classif, RFECV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(filepath='churn_clean.csv'):
    """
    Load the dataset and remove irrelevant variables
    """
    df = pd.read_csv(filepath)
    
    # Step 1: Remove Irrelevant Variables
    irrelevant_columns = [
        # Customer Identification Variables
        'Customer_id', 'CaseOrder', 'UID', 'Interaction',
        # Granular Geographic Data
        'Lat', 'Lng', 'Zip', 'City', 'County',
        # Highly Correlated with Tenure
        'Bandwidth_GB_Year'
    ]
    return df.drop(columns=irrelevant_columns)

def encode_features(df):
    """
    Encode boolean and categorical features with a mix of techniques
    """
    df_encoded = df.copy()
    
    # Convert boolean-like strings to 1/0
    bool_columns = ['Churn', 'Techie', 'Port_modem', 'Tablet', 'Phone', 'Multiple', 
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                   'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    for col in bool_columns:
        df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
    
    # Label encode high-cardinality features
    high_cardinality = ['State', 'Job', 'TimeZone']
    label_encoder = LabelEncoder()
    for col in high_cardinality:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
    
    # One-hot encode remaining categorical variables
    remaining_categorical = ['Marital', 'Gender', 'InternetService', 
                           'Contract', 'PaymentMethod', 'Area']
    df_encoded = pd.get_dummies(df_encoded, columns=remaining_categorical, drop_first=True)
    
    return df_encoded

def analyze_feature_importance(X, y):
    """
    Analyze feature importance using multiple methods
    """
    print("\nStarting Mutual Information calculation...")
    # Calculate Mutual Information scores
    mi_scores = mutual_info_classif(X, y)
    mi_scores_dict = dict(zip(X.columns, mi_scores))
    mi_scores_sorted = {k: v for k, v in sorted(mi_scores_dict.items(), key=lambda item: item[1], reverse=True)}
    print("✓ Mutual Information calculation complete")
    
    # Filter out bottom 16 features based on MI scores
    features_to_keep = list(mi_scores_sorted.keys())[:-16]  # Keep all except bottom 16
    X_filtered = X[features_to_keep]
    print(f"\nRemoved 16 lowest MI score features. Continuing with {len(features_to_keep)} features")
    
    print("\nStarting Recursive Feature Elimination (this may take a while)...")
    # Recursive Feature Elimination
    rfe = RFECV(
        estimator=XGBClassifier(
            random_state=42,
            n_jobs=-1,
            n_estimators=100
        ),
        step=5,
        cv=10,
        scoring='accuracy',
        n_jobs=-1,
        min_features_to_select=10
    )
    rfe.fit(X_filtered, y)
    selected_features = X_filtered.columns[rfe.support_].tolist()
    print("✓ Recursive Feature Elimination complete")
    
    print("\nFeature analysis completed successfully!")
    return {
        'mi_scores': mi_scores_sorted,
        'selected_features': selected_features,
        'optimal_n_features': rfe.n_features_,
        'removed_features': list(mi_scores_sorted.keys())[-16:]
    }

def plot_feature_analysis(analysis_results, X):
    """
    Create and save visualization plots
    """
    # Get filtered features
    remaining_features = list(analysis_results['mi_scores'].keys())[:-16]
    X_filtered = X[remaining_features]
    
    # Mutual Information Plot
    plt.figure(figsize=(12, 6))
    mi_df = pd.DataFrame(list(analysis_results['mi_scores'].items()), 
                        columns=['Feature', 'Mutual_Information'])
    sns.barplot(data=mi_df, x='Mutual_Information', y='Feature')
    plt.title('Features by Mutual Information Score')
    plt.tight_layout()
    plt.savefig('mutual_information.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_summary(analysis_results, X):
    """
    Create a summary DataFrame of all feature selection methods
    """
    # Get the features that weren't removed (all features except the bottom 16)
    remaining_features = list(analysis_results['mi_scores'].keys())[:-16]
    
    feature_summary = pd.DataFrame({
        'Feature': remaining_features,
        'Mutual_Info_Score': [analysis_results['mi_scores'][feat] for feat in remaining_features],
        'Selected_by_RFECV': [feat in analysis_results['selected_features'] for feat in remaining_features]
    })
    return feature_summary.sort_values('Mutual_Info_Score', ascending=False)

def print_analysis_results(analysis_results):
    """
    Print the analysis results in a formatted way
    """
    print("\n=== Feature Selection Results ===")
    
    print("\nFeatures Removed Based on Low Mutual Information:")
    for feature in analysis_results['removed_features']:
        print(f"{feature}: {analysis_results['mi_scores'][feature]:.4f}")
    
    print("\nTop 10 Features by Mutual Information:")
    for feature, score in list(analysis_results['mi_scores'].items())[:10]:
        print(f"{feature}: {score:.4f}")
    
    print("\nOptimal number of features (RFECV):", analysis_results['optimal_n_features'])
    print("\nSelected Features by RFECV:")
    print(analysis_results['selected_features'])

def get_selected_features(df, analysis_results):
    """
    Return the dataset with only the selected features
    """
    return df[analysis_results['selected_features'] + ['Churn']]

def main():
    print("\nStarting preprocessing pipeline...")
    # Load and clean data
    print("\nStep 1: Loading and cleaning data...")
    df_cleaned = load_and_clean_data()
    print("✓ Data loaded and cleaned")
    
    # Encode features
    print("\nStep 2: Encoding features...")
    df_encoded = encode_features(df_cleaned)
    print("✓ Features encoded")
    
    # Separate features and target
    print("\nStep 3: Separating features and target...")
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    print("✓ Features and target separated")
    
    # Perform feature analysis
    print("\nStep 4: Performing feature analysis (this may take several minutes)...")
    analysis_results = analyze_feature_importance(X, y)
    
    # Create visualizations
    print("\nStep 5: Creating visualizations...")
    plot_feature_analysis(analysis_results, X)
    print("✓ Visualizations created")
    
    # Create and save feature summary
    print("\nStep 6: Creating feature summary...")
    feature_summary = create_feature_summary(analysis_results, X)
    feature_summary.to_csv('feature_selection_summary.csv', index=False)
    print("✓ Feature summary saved to CSV")
    
    # Print results
    print("\nStep 7: Printing analysis results...")
    print_analysis_results(analysis_results)
    
    # Get final selected features dataset
    print("\nStep 8: Creating final dataset...")
    final_df = get_selected_features(df_encoded, analysis_results)
    print("✓ Final dataset created")
    
    return final_df, analysis_results

if __name__ == "__main__":
    final_df, analysis_results = main()
    print("\nPreprocessing complete. Final dataset shape:", final_df.shape)
    print("Detailed feature selection summary saved to 'feature_selection_summary.csv'")
