import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Set plot style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Create directories for outputs
model_dir = '/data/chats/iixf8/workspace/models'
visual_dir = '/data/chats/iixf8/workspace/visualizations'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(visual_dir, exist_ok=True)

print("Building and evaluating machine learning models for churn prediction...\n")

# 1. Load the preprocessed dataset
df = pd.read_csv('/data/chats/iixf8/workspace/telco_churn_processed.csv')
print(f"Loaded preprocessed dataset with shape: {df.shape}")

# 2. Data preparation and feature selection
# Remove customerID as it's not a predictive feature
df = df.drop('customerID', axis=1)

# Display target variable distribution
print(f"\nTarget variable 'Churn' distribution:\n{df['Churn'].value_counts()}")
print(f"Churn rate: {df['Churn_Binary'].mean():.4f}")

# 3. Feature selection
# Based on previous analysis, we'll use both categorical and numerical features
# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')  # Remove target from features

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'CLV']

# Print selected features
print(f"\nSelected categorical features ({len(categorical_cols)}):\n{categorical_cols}")
print(f"\nSelected numerical features ({len(numerical_cols)}):\n{numerical_cols}")

# 4. Split the data into training and test sets
X = df.drop(['Churn', 'Churn_Binary'], axis=1)  # Features
y = df['Churn_Binary']  # Target: binary churn indicator

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 5. Define preprocessing steps for numerical and categorical features
# Numeric features: impute missing values and standardize
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical features: impute missing values and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 6. Define evaluation metrics function
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probability predictions for ROC and PR curves
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    avg_precision = average_precision_score(y_test, y_test_proba)
    
    print(f"\n--- {model_name} Performance ---")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'{visual_dir}/{model_name}_confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'{visual_dir}/{model_name}_roc_curve.png')
    plt.close()
    
    # Detailed classification report
    print(f"\nClassification Report:\n{classification_report(y_test, y_test_pred)}")
    
    return {
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }

# 7. Build and evaluate different models
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('XGBoost', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
]

results = []

for model_name, model in models:
    print(f"\nTraining {model_name}...")
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    result = evaluate_model(pipeline, X_train, X_test, y_train, y_test, model_name)
    results.append(result)

# 8. Compare model performances
results_df = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
print("\n--- Model Performance Comparison ---")
print(results_df.sort_values('f1_score', ascending=False).reset_index(drop=True))

# 9. Visualize model comparison
plt.figure(figsize=(14, 10))

# F1 Score comparison
plt.subplot(2, 2, 1)
sns.barplot(x='model_name', y='f1_score', data=results_df)
plt.title('F1 Score by Model')
plt.xticks(rotation=45)

# ROC AUC comparison
plt.subplot(2, 2, 2)
sns.barplot(x='model_name', y='roc_auc', data=results_df)
plt.title('ROC AUC by Model')
plt.xticks(rotation=45)

# Precision comparison
plt.subplot(2, 2, 3)
sns.barplot(x='model_name', y='precision', data=results_df)
plt.title('Precision by Model')
plt.xticks(rotation=45)

# Recall comparison
plt.subplot(2, 2, 4)
sns.barplot(x='model_name', y='recall', data=results_df)
plt.title('Recall by Model')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f'{visual_dir}/model_comparison.png')
plt.close()

# 10. Select the best model (based on F1 score)
best_model_idx = results_df['f1_score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'model_name']
best_model = [model for name, model in models if name == best_model_name][0]

print(f"\nBest model based on F1 score: {best_model_name}")

# 11. Feature Importance for the best model (if applicable)
if best_model_name in ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'XGBoost']:
    # Rebuild the best model pipeline
    best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', best_model)])
    best_pipeline.fit(X_train, y_train)
    
    # Get feature names from the preprocessor
    feature_names = []
    
    # Numerical feature names remain the same
    for feature in numerical_cols:
        feature_names.append(feature)
    
    # Get one-hot encoded categorical feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    for i, category in enumerate(categorical_cols):
        for value in ohe.categories_[i]:
            feature_names.append(f"{category}_{value}")
    
    # Get feature importances from the model
    importances = best_pipeline.named_steps['model'].feature_importances_
    
    # Create feature importance DataFrame
    if len(feature_names) == len(importances):
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Print top 20 most important features
        print("\n--- Top 20 Most Important Features ---")
        print(feature_importance_df.head(20))
        
        # Visualize feature importance
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
        plt.title(f'Top 20 Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig(f'{visual_dir}/{best_model_name}_feature_importance.png')
        plt.close()
    else:
        print("\nWarning: Feature names and importance scores don't match in length.")

# 12. Save model performance summary
rows = []
for _, row in results_df.iterrows():
    model_row = f"| {row['model_name']} | {row['test_accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {row['roc_auc']:.4f} |"
    rows.append(model_row)

model_summary = f"""# Telecom Customer Churn Prediction: Model Evaluation

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|--------|
{chr(10).join(rows)}

## Best Performing Model

The best performing model was **{best_model_name}** with an F1 score of {results_df.loc[best_model_idx, 'f1_score']:.4f}.

## Model Performance Insights

- **Precision**: {results_df.loc[best_model_idx, 'precision']:.4f} - This indicates that when the model predicts a customer will churn, it is correct about {results_df.loc[best_model_idx, 'precision']*100:.1f}% of the time.

- **Recall**: {results_df.loc[best_model_idx, 'recall']:.4f} - The model can identify {results_df.loc[best_model_idx, 'recall']*100:.1f}% of all customers who will actually churn.

- **F1 Score**: {results_df.loc[best_model_idx, 'f1_score']:.4f} - This is the harmonic mean of precision and recall, providing a balanced measure of model performance.

- **ROC AUC**: {results_df.loc[best_model_idx, 'roc_auc']:.4f} - This indicates good discrimination ability between churned and non-churned customers.

## Business Implications

1. **Early Intervention**: The model can identify {results_df.loc[best_model_idx, 'recall']*100:.1f}% of customers who are likely to churn, allowing for targeted retention strategies.

2. **Efficient Resource Allocation**: With a precision of {results_df.loc[best_model_idx, 'precision']:.4f}, retention efforts can be focused on customers who are most likely to churn, improving the ROI of retention campaigns.

3. **Customer Lifetime Value Preservation**: By identifying at-risk high-value customers early, the company can prioritize retention efforts to preserve customer lifetime value.

## Recommendations for Implementation

1. **Deployment Strategy**: Integrate the churn prediction model into the customer management system to flag at-risk customers on a monthly basis.

2. **Threshold Tuning**: Consider adjusting the probability threshold based on business priorities (higher recall vs. higher precision).

3. **Segment-Specific Models**: Consider developing separate models for different customer segments (e.g., new vs. long-term customers) to further improve prediction accuracy.

4. **Continuous Monitoring**: Implement a monitoring system to track model performance over time and retrain as needed to maintain prediction accuracy.

5. **A/B Testing**: Test different intervention strategies on predicted churners to identify the most effective retention approaches.
"""

with open(f'{model_dir}/model_performance_summary.md', 'w') as f:
    f.write(model_summary)

print(f"\nModel evaluation complete. Results saved to {model_dir}/model_performance_summary.md")
print(f"Visualizations saved to {visual_dir}/")
