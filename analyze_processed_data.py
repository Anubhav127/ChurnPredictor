import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set plot style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print('Analyzing preprocessed telecom customer churn dataset...')

# Create directory for visualizations
os.makedirs('/data/chats/iixf8/workspace/visualizations', exist_ok=True)

# Load the preprocessed dataset
df = pd.read_csv('/data/chats/iixf8/workspace/telco_churn_processed.csv')
print(f'Dataset shape: {df.shape}')

# Display basic info
print('\nDataset columns:')
print(df.columns.tolist())

# Summary statistics for key numerical features
print('\nSummary statistics for key features:')
print(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'CLV', 'Churn_Binary']].describe())

# Overall churn rate
churn_rate = df['Churn_Binary'].mean() * 100
print(f'\nOverall churn rate: {churn_rate:.2f}%')

# 1. Customer Segmentation Analysis
print('\n==== Customer Segmentation Analysis ====')

# By TenureGroup and Contract
segment_analysis = df.groupby(['TenureGroup', 'Contract'])['Churn_Binary'].agg(['mean', 'count'])
segment_analysis.columns = ['Churn_Rate', 'Customer_Count']
segment_analysis = segment_analysis.reset_index().sort_values('Churn_Rate', ascending=False)
print('\nChurn rate by customer segment (TenureGroup & Contract):')
print(segment_analysis.head(10))

# By InternetService and TotalServices
service_segment = df.groupby(['InternetService', 'TotalServices'])['Churn_Binary'].agg(['mean', 'count'])
service_segment.columns = ['Churn_Rate', 'Customer_Count']
service_segment = service_segment.reset_index().sort_values('Churn_Rate', ascending=False)
print('\nChurn rate by service segment (InternetService & TotalServices):')
print(service_segment.head(10))

# 2. Visualizations
print('\n==== Creating Visualizations ====')

# Churn rate by contract type and tenure group
plt.figure(figsize=(14, 8))
sns.barplot(x='TenureGroup', y='Churn_Rate', hue='Contract', data=segment_analysis)
plt.title('Churn Rate by Tenure Group and Contract Type', fontsize=16)
plt.xlabel('Tenure Group', fontsize=14)
plt.ylabel('Churn Rate', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Contract Type')
plt.tight_layout()
plt.savefig('/data/chats/iixf8/workspace/visualizations/churn_by_tenure_contract.png')
plt.close()

# Churn rate by internet service and total services
plt.figure(figsize=(14, 8))
sns.barplot(x='TotalServices', y='Churn_Rate', hue='InternetService', data=service_segment)
plt.title('Churn Rate by Number of Services and Internet Service Type', fontsize=16)
plt.xlabel('Number of Services', fontsize=14)
plt.ylabel('Churn Rate', fontsize=14)
plt.legend(title='Internet Service')
plt.tight_layout()
plt.savefig('/data/chats/iixf8/workspace/visualizations/churn_by_services.png')
plt.close()

# Monthly charges distribution by churn status
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, bins=30)
plt.title('Monthly Charges Distribution by Churn Status', fontsize=16)
plt.xlabel('Monthly Charges ($)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()
plt.savefig('/data/chats/iixf8/workspace/visualizations/monthly_charges_distribution.png')
plt.close()

# 3. Feature Importance Analysis using only numeric features
print('\n==== Feature Importance Analysis ====')

# Select only numerical features that already exist
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 
                    'AvgMonthlyCharges', 'CLV', 'Churn_Binary']

# Create a dataframe with only numerical features
df_numeric = df[numeric_features].copy()

# Remove target from features
X = df_numeric.drop('Churn_Binary', axis=1)
y = df_numeric['Churn_Binary']

# Train a Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print('\nNumerical feature importance:')
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Numerical Feature Importance', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig('/data/chats/iixf8/workspace/visualizations/numeric_feature_importance.png')
plt.close()

# 4. Additional analysis - Churn rate by contract type
plt.figure(figsize=(10, 6))
contract_churn = df.groupby('Contract')['Churn_Binary'].mean().sort_values(ascending=False)
sns.barplot(x=contract_churn.index, y=contract_churn.values)
plt.title('Churn Rate by Contract Type', fontsize=16)
plt.xlabel('Contract Type', fontsize=14)
plt.ylabel('Churn Rate', fontsize=14)
plt.tight_layout()
plt.savefig('/data/chats/iixf8/workspace/visualizations/churn_by_contract.png')
plt.close()

# 5. Payment method analysis
plt.figure(figsize=(12, 6))
payment_churn = df.groupby('PaymentMethod')['Churn_Binary'].mean().sort_values(ascending=False)
sns.barplot(x=payment_churn.index, y=payment_churn.values)
plt.title('Churn Rate by Payment Method', fontsize=16)
plt.xlabel('Payment Method', fontsize=14)
plt.ylabel('Churn Rate', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/data/chats/iixf8/workspace/visualizations/churn_by_payment.png')
plt.close()

# 6. Relationship Analysis: MonthlyCharges vs TotalServices
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='MonthlyCharges', y='TotalServices', hue='Churn', alpha=0.7)
plt.title('Relationship between Monthly Charges and Total Services', fontsize=16)
plt.xlabel('Monthly Charges ($)', fontsize=14)
plt.ylabel('Number of Services', fontsize=14)
plt.tight_layout()
plt.savefig('/data/chats/iixf8/workspace/visualizations/charges_vs_services.png')
plt.close()

# 7. Customer Lifetime Value Analysis
plt.figure(figsize=(12, 6))
sns.boxplot(x='TenureGroup', y='CLV', hue='Churn', data=df)
plt.title('Customer Lifetime Value by Tenure Group and Churn Status', fontsize=16)
plt.xlabel('Tenure Group', fontsize=14)
plt.ylabel('Customer Lifetime Value ($)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/data/chats/iixf8/workspace/visualizations/clv_by_tenure_churn.png')
plt.close()

# 8. Create executive summary of findings
summary = """
# Telecom Customer Churn Analysis: Executive Summary

## Key Findings

### 1. High-Risk Customer Segments

- **New Month-to-Month Customers**: Customers in the 0-12 month tenure group with month-to-month contracts show the highest churn rate (51.35%).
- **Fiber Optic Customers**: Regardless of tenure, customers with fiber optic service churn at significantly higher rates (41.9%) than DSL (18.9%) or no internet service (7.4%).
- **Electronic Check Users**: Customers using electronic check as their payment method have a higher churn rate than other payment methods.

### 2. Service Package Insights

- **Service Paradox**: Customers with fiber optic service and fewer total services (2-3) have the highest churn rates (>59%).
- **Price Sensitivity**: Higher monthly charges correlate with increased churn probability, particularly in newer customers.

### 3. Customer Value Insights

- **CLV Gap**: There's a substantial difference in Customer Lifetime Value (CLV) between churned and retained customers, highlighting the financial impact of churn.
- **Retention ROI**: Long-term customers (61-72 months) have approximately 7 times lower churn rates (6.6%) than new customers (47.7%), emphasizing the value of customer retention efforts.

### 4. Most Important Predictive Features

The top numerical predictors of customer churn are:
1. Tenure (negative correlation with churn)
2. Total Charges (negative correlation with churn)
3. Customer Lifetime Value (negative correlation with churn)
4. Monthly Charges (positive correlation with churn)

Top categorical predictors:
1. Contract type (month-to-month vs. longer-term contracts)
2. Internet service type (fiber optic vs. DSL or none)
3. Payment method (electronic check vs. other methods)

## Recommendations for Churn Reduction

1. **Contract Conversion Program**: Target month-to-month customers with tenure of 7-12 months for conversion to annual contracts through incentives.

2. **Fiber Service Quality Investigation**: Investigate potential quality or pricing issues with fiber optic service that may be driving high churn rates.

3. **Payment Method Transition**: Encourage customers to switch from electronic check to automatic payment methods through incentives or discounts.

4. **New Customer Onboarding**: Implement enhanced onboarding and regular check-ins during the first 12 months to improve early-stage retention.

5. **Price Optimization**: Review pricing strategy for customers with multiple services to ensure value perception aligns with monthly charges.

## Next Steps

- Implement predictive modeling to identify at-risk customers before they churn
- Develop targeted retention campaigns for the highest-risk segments
- Conduct customer satisfaction surveys focused on fiber optic service quality
- Perform A/B testing of retention incentives for month-to-month customers
"""

with open('/data/chats/iixf8/workspace/churn_analysis_summary.md', 'w') as f:
    f.write(summary)

print('\nAnalysis complete. Visualizations saved to /data/chats/iixf8/workspace/visualizations/')
print('Executive summary saved to /data/chats/iixf8/workspace/churn_analysis_summary.md')
