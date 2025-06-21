
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
