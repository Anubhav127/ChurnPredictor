
# Telco Customer Churn Data Preprocessing and Feature Engineering Report

## Dataset Overview
- Original shape: (7043, 21)
- Processed shape: (7043, 28)

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
Churn_Binary         1.000000
MonthlyCharges       0.193356
AvgMonthlyCharges    0.192531
TotalServices       -0.019375
TotalCharges        -0.198324
CLV                 -0.198514
tenure              -0.352229

### Categorical Features with Highest Churn Rates:
- Contract: Month-to-month contracts have the highest churn rate
- Internet Service: Fiber optic customers churn more frequently
- Tenure Group: Customers with 0-12 months tenure have highest churn
- Payment Method: Electronic check users show higher churn propensity

## Recommendations for Modeling
1. Use both original and engineered features for modeling
2. Consider applying class weighting or resampling to address class imbalance (26.54% churn rate)
3. Focus on Contract type, TenureGroup, and InternetService as they show strong associations with churn
