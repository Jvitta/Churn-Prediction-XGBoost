import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('churn_clean.csv')

# Identify columns to exclude (identifiers and redundant columns)
exclude_columns = ['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'County', 'Zip', 'Job']

# Create a copy of the dataframe without excluded columns
df_analysis = df.drop(columns=exclude_columns)

# Convert boolean-like strings to 1/0
bool_columns = ['Churn', 'Techie', 'Port_modem', 'Tablet', 'Phone', 'Multiple', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
for col in bool_columns:
    df_analysis[col] = (df_analysis[col] == 'Yes').astype(int)

# One-hot encode remaining categorical variables
categorical_columns = ['State', 'TimeZone', 'Marital', 'Gender', 'InternetService', 
                      'Contract', 'PaymentMethod', 'Area']
df_encoded = pd.get_dummies(df_analysis, columns=categorical_columns, drop_first=True)

# Calculate correlation matrix
correlation_matrix = df_encoded.corr()

# Create a larger figure for better readability
plt.figure(figsize=(20, 16))

# Create a heatmap
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme
            center=0,  # Center the colormap at 0
            fmt='.2f',  # Format correlation values to 2 decimal places
            square=True)  # Make the plot square-shaped

plt.title('Correlation Heatmap (Including Encoded Categorical Variables)')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()  # Adjust layout to prevent label cutoff

# Save the plot
plt.savefig('correlation_heatmap_with_categories.png', dpi=300, bbox_inches='tight')
plt.close()

print("Enhanced correlation heatmap has been saved as 'correlation_heatmap_with_categories.png'")

# Print the strongest correlations with Churn
churn_correlations = correlation_matrix['Churn'].sort_values(ascending=False)
print("\nTop 10 features most correlated with Churn:")
print(churn_correlations[1:11])  # Excluding Churn's correlation with itself