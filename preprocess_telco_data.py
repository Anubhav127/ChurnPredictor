import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

print('Starting data preprocessing and feature engineering...')

# Load the dataset
df = pd.read_csv('/data/chats/iixf8/workspace/telco_customer_churn.csv')
print(f'Dataset shape: {df.shape}')

# Make a copy to preserve original data
df_processed = df.copy()

# 1. Handle missing values
# Convert TotalCharges to numeric
df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')

# Check for missing values after conversion
print('\nMissing values after TotalCharges conversion:')
missing_values = df_processed.isnull().sum()
print(missing_values[missing_values > 0])

# Investigate the records with missing TotalCharges
print('\nRecords with missing TotalCharges:')
print(df[df_processed['TotalCharges'].isnull()])

# Since all customers with missing TotalCharges have tenure=0, we can impute with 0
df_processed['TotalCharges'].fillna(0, inplace=True)
print('\nMissing values after imputation:')
print(df_processed.isnull().sum().sum())

# 2. Convert SeniorCitizen to categorical
df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

# 3. Feature Engineering

# Create tenure groups
bins = [0, 12, 24, 36, 48, 60, 72]
labels = ['0-12 months', '13-24 months', '25-36 months', '37-48 months', '49-60 months', '61-72 months']
df_processed['TenureGroup'] = pd.cut(df_processed['tenure'], bins=bins, labels=labels)

# Create total services feature
service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Function to count services
def count_services(row):
    count = 0
    # Count phone service
    if row['PhoneService'] == 'Yes':
        count += 1
        if row['MultipleLines'] == 'Yes':
            count += 1
    # Count internet service and related services
    if row['InternetService'] != 'No':
        count += 1  # Count the internet service itself
        for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
            if row[col] == 'Yes':
                count += 1
    return count

df_processed['TotalServices'] = df_processed.apply(count_services, axis=1)

# Create average monthly charge feature
df_processed['AvgMonthlyCharges'] = df_processed.apply(
    lambda row: row['MonthlyCharges'] if row['tenure'] <= 1 else row['TotalCharges'] / row['tenure'], 
    axis=1
)

# Create categoricals for MonthlyCharges
df_processed['MonthlyChargeCategory'] = pd.qcut(df_processed['MonthlyCharges'], 4, labels=['Low', 'Medium', 'High', 'Very High'])

# Create recency indicator (if data were time-stamped)
# Here we'll use tenure as a proxy for recency
df_processed['Customer_Status'] = df_processed['tenure'].apply(
    lambda x: 'New' if x <= 12 else 'Intermediate' if x <= 36 else 'Established'
)

# Create customer lifetime value (simplified version)
df_processed['CLV'] = df_processed['tenure'] * df_processed['MonthlyCharges']

# 4. Prepare for modeling

# Identify categorical and numerical features
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                       'TenureGroup', 'MonthlyChargeCategory', 'Customer_Status']

numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'AvgMonthlyCharges', 'CLV']

# Create binary target variable (0: No churn, 1: Yes churn)
df_processed['Churn_Binary'] = df_processed['Churn'].map({'No': 0, 'Yes': 1})

# 5. Feature Analysis and Visualization

# Correlation analysis for numerical features
numerical_data = df_processed[numerical_features + ['Churn_Binary']]
corr_matrix = numerical_data.corr()

# Plot correlation matrix
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('/data/chats/iixf8/workspace/correlation_matrix.png')
plt.close()

# Feature importance based on correlation with target
feature_corr_with_target = corr_matrix['Churn_Binary'].sort_values(ascending=False)
print('\nFeature correlation with Churn:')
print(feature_corr_with_target)

# Analyze categorical features
print('\nChurn rate by key categorical features:')
for feature in ['Contract', 'InternetService', 'TenureGroup', 'PaymentMethod']:
    churn_by_feature = df_processed.groupby(feature)['Churn_Binary'].mean().sort_values(ascending=False)
    print(f'\n{feature}:')
    print(churn_by_feature)
    
    # Plot churn rate by feature
    plt.figure(figsize=(10, 6))
    sns.barplot(x=churn_by_feature.index, y=churn_by_feature.values)
    plt.title(f'Churn Rate by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'/data/chats/iixf8/workspace/churn_by_{feature}.png')
    plt.close()

# 6. Save processed data
df_processed.to_csv('/data/chats/iixf8/workspace/telco_churn_processed.csv', index=False)
print('\nProcessed data saved to telco_churn_processed.csv')

# 7. Generate processing report
report = f"""
# Telco Customer Churn Data Preprocessing and Feature Engineering Report

## Dataset Overview
- Original shape: {df.shape}
- Processed shape: {df_processed.shape}

## Preprocessing Steps Performed
1. Handled missing values in TotalCharges (imputed with 0 for customers with tenure=0)
2. Converted SeniorCitizen from numeric (0/1) to categorical (No/Yes)

## Feature Engineering

New features created:
1. **TenureGroup**: Categorized tenure into meaningful groups
2. **TotalServices**: Count of services subscribed by each customer
3. **AvgMonthlyCharges**: Total charges divided by tenure
4. **MonthlyChargeCategory**: Categorized monthly charges into quartiles
5. **Customer_Status**: Classification based on tenure (New/Intermediate/Established)
6. **CLV**: Simple Customer Lifetime Value calculation (tenure * MonthlyCharges)
7. **Churn_Binary**: Numeric representation of Churn (0=No, 1=Yes)

## Key Feature Insights

### Top Numerical Feature Correlations with Churn:
{feature_corr_with_target.to_string()}

### Categorical Features with Highest Churn Rates:
- Contract: Month-to-month contracts have the highest churn rate
- Internet Service: Fiber optic customers churn more frequently
- Tenure Group: Customers with 0-12 months tenure have highest churn
- Payment Method: Electronic check users show higher churn propensity

## Recommendations for Modeling
1. Use both original and engineered features for modeling
2. Consider applying class weighting or resampling to address class imbalance (26.54% churn rate)
3. Focus on Contract type, TenureGroup, and InternetService as they show strong associations with churn
"""

with open('/data/chats/iixf8/workspace/preprocessing_report.md', 'w') as f:
    f.write(report)

print('\nPreprocessing report saved to preprocessing_report.md')
print('\nData preprocessing and feature engineering complete!')
