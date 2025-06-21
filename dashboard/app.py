import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import joblib
from PIL import Image
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Set page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the data
@st.cache_data
def load_data():
    # Construct the absolute path to the data file
    data_path = os.path.join(os.path.dirname(__file__), '..', 'telco_churn_processed.csv')
    df = pd.read_csv(data_path)
    return df

# Build the model pipeline
@st.cache_resource
def build_model():
    # Define numerical and categorical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'CLV']
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                        'PhoneService', 'MultipleLines', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 
                        'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Create the model pipeline with the best model (Logistic Regression)
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', LogisticRegression(max_iter=1000, random_state=42))])
    
    # Train with all data
    df = load_data()
    X = df.drop(['customerID', 'Churn', 'Churn_Binary'], axis=1)
    y = df['Churn_Binary']
    model.fit(X, y)
    
    return model, numerical_cols, categorical_cols

# Define a function to generate retention recommendations
def get_retention_recommendations(prediction, prediction_proba, customer_data):
    # If prediction is churn (1)
    if prediction == 1:
        risk_level = "High" if prediction_proba > 0.7 else "Medium"
        
        recommendations = []
        
        # Contract-based recommendations
        if customer_data.get('Contract') == 'Month-to-month':
            recommendations.append("Offer a discounted annual contract with a loyalty bonus.")
        
        # Internet service recommendations
        if customer_data.get('InternetService') == 'Fiber optic':
            recommendations.append("Consider discussing service quality issues and offer a free service review.")
        
        # Payment method recommendations
        if customer_data.get('PaymentMethod') == 'Electronic check':
            recommendations.append("Encourage transition to automatic payment methods with a one-time discount.")
        
        # Tenure-based recommendations
        if customer_data.get('tenure') <= 12:
            recommendations.append("Implement enhanced onboarding and schedule regular check-ins.")
        
        # Service-based recommendations
        if customer_data.get('TotalServices') <= 3:
            recommendations.append("Offer a bundled services package with a promotional rate.")
        
        # Price optimization
        if customer_data.get('MonthlyCharges') > 70:
            recommendations.append("Review pricing strategy and offer a personalized discount.")
            
        # If no specific recommendations were generated, add a generic one
        if not recommendations:
            recommendations.append("Contact customer for a personalized retention offer.")
        
        return {
            "risk_level": risk_level,
            "recommendations": recommendations
        }
    else:
        # For non-churning prediction, return nurturing recommendations
        return {
            "risk_level": "Low",
            "recommendations": ["Continue regular customer engagement and satisfaction monitoring."]
        }

# Create a function to generate charts for the overview page
def create_overview_charts(df):
    # 1. Churn Distribution Pie Chart
    churn_counts = df['Churn'].value_counts()
    fig1 = px.pie(
        values=churn_counts.values, 
        names=churn_counts.index,
        title='Customer Churn Distribution',
        color_discrete_sequence=['#3498db', '#e74c3c'],
        labels={'label': 'Churn Status', 'value': 'Count'},
        hole=0.4
    )
    
    # 2. Churn by Contract Type
    contract_churn = df.groupby(['Contract', 'Churn']).size().unstack().fillna(0)
    contract_churn_pct = (contract_churn['Yes'] / (contract_churn['Yes'] + contract_churn['No']) * 100).reset_index()
    contract_churn_pct.columns = ['Contract', 'Churn Rate (%)']    
    fig2 = px.bar(
        contract_churn_pct, 
        x='Contract', 
        y='Churn Rate (%)',
        title='Churn Rate by Contract Type',
        color='Churn Rate (%)',
        color_continuous_scale=['#3498db', '#e74c3c']
    )
    
    # 3. Churn by Tenure Group
    tenure_churn = df.groupby(['TenureGroup', 'Churn']).size().unstack().fillna(0)
    tenure_churn_pct = (tenure_churn['Yes'] / (tenure_churn['Yes'] + tenure_churn['No']) * 100).reset_index()
    tenure_churn_pct.columns = ['Tenure Group', 'Churn Rate (%)']
    fig3 = px.bar(
        tenure_churn_pct, 
        x='Tenure Group', 
        y='Churn Rate (%)',
        title='Churn Rate by Tenure Group',
        color='Churn Rate (%)',
        color_continuous_scale=['#3498db', '#e74c3c']
    )
    
    # 4. Monthly Charges vs Churn
    fig4 = px.histogram(
        df, 
        x='MonthlyCharges',
        color='Churn',
        barmode='group',
        title='Monthly Charges Distribution by Churn Status',
        color_discrete_sequence=['#3498db', '#e74c3c']
    )
    
    return fig1, fig2, fig3, fig4

# Create app header
def render_header():
    st.title("📊 Telecom Customer Churn Prediction Dashboard")
    st.markdown("""
    This dashboard provides insights into customer churn patterns and allows you to predict customer churn risk.
    Use the sidebar to navigate between different sections of the dashboard.
    """)

# Create overview page
def overview_page(df):
    st.header("Customer Churn Overview")
    
    # Key metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(df):,}"
        )
    
    with col2:
        churn_rate = df['Churn_Binary'].mean() * 100
        st.metric(
            label="Overall Churn Rate",
            value=f"{churn_rate:.2f}%"
        )
    
    with col3:
        avg_tenure = df['tenure'].mean()
        st.metric(
            label="Avg. Customer Tenure",
            value=f"{avg_tenure:.1f} months"
        )
    
    with col4:
        avg_monthly = df['MonthlyCharges'].mean()
        st.metric(
            label="Avg. Monthly Charge",
            value=f"${avg_monthly:.2f}"
        )
    
    # Charts
    fig1, fig2, fig3, fig4 = create_overview_charts(df)
    
    st.subheader("Churn Distribution and Key Factors")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Key findings
    st.subheader("Key Findings")
    st.markdown("""
    - **Contract Impact**: Month-to-month customers have a drastically higher churn rate compared to those with annual or two-year contracts.
    - **Tenure Matters**: New customers (0-12 months) are much more likely to churn than established customers.
    - **Price Sensitivity**: Higher monthly charges correlate with increased churn probability.
    - **Service Quality**: Customers with Fiber Optic internet service show higher churn rates despite the premium service.
    """)

# Create model performance page
def model_performance_page():
    st.header("Model Performance Metrics")
    
    # Display model comparison metrics
    st.subheader("Model Performance Comparison")
    
    # Create a dataframe with model comparison data (from the model performance summary)
    model_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
        'Accuracy': [0.8048, 0.7339, 0.7814, 0.8006, 0.7814],
        'Precision': [0.6667, 0.4987, 0.6107, 0.6598, 0.6012],
        'Recall': [0.5294, 0.5134, 0.4866, 0.5134, 0.5241],
        'F1 Score': [0.5902, 0.5059, 0.5417, 0.5774, 0.5600],
        'ROC AUC': [0.8429, 0.6632, 0.8214, 0.8432, 0.8238]
    }
    
    model_df = pd.DataFrame(model_data)
    st.dataframe(model_df, use_container_width=True)
    
    # Model performance visualizations
    st.subheader("Model Performance Visualizations")
    
    # Display model comparison chart
    try:
        img_path = os.path.join(os.path.dirname(__file__), 'assets', 'model_comparison.png')
        comparison_img = Image.open(img_path)
        st.image(comparison_img, caption="Model Performance Metrics Comparison", use_container_width=True)
    except:
        st.warning("Model comparison visualization not available")
    
    # Display confusion matrix for the best model
    try:
        col1, col2 = st.columns(2)
        with col1:
            cm_path = os.path.join(os.path.dirname(__file__), 'assets', 'Logistic Regression_confusion_matrix.png')
            confusion_img = Image.open(cm_path)
            st.image(confusion_img, caption="Confusion Matrix - Logistic Regression", use_container_width=True)
        
        with col2:
            roc_path = os.path.join(os.path.dirname(__file__), 'assets', 'Logistic Regression_roc_curve.png')
            roc_img = Image.open(roc_path)
            st.image(roc_img, caption="ROC Curve - Logistic Regression", use_container_width=True)
    except:
        st.warning("Model performance visualizations not available")
    
    # Model insights
    st.subheader("Model Insights")
    st.markdown("""
    **The best performing model is Logistic Regression with:**
    - **Precision**: 0.6667 - When the model predicts a customer will churn, it is correct 66.7% of the time.
    - **Recall**: 0.5294 - The model identifies 52.9% of all customers who will actually churn.
    - **F1 Score**: 0.5902 - Balanced measure of model performance.
    - **ROC AUC**: 0.8429 - Good discrimination ability between churned and non-churned customers.
    
    **Business Implementation:**
    - Integrating this model would allow the company to identify more than half of potential churners.
    - With a precision of 66.7%, targeted retention campaigns would be efficient.
    - The model's good ROC AUC score indicates reliable classification performance.
    """)

# Create prediction page
def prediction_page(model, num_cols, cat_cols):
    st.header("Customer Churn Prediction")
    st.markdown("Enter customer information to predict churn probability")
    
    # Create a form for customer data input
    with st.form("prediction_form"):
        # Divide form into multiple columns
        col1, col2, col3 = st.columns(3)
        
        # Personal Information
        with col1:
            st.subheader("Personal Information")
            gender = st.selectbox("Gender", options=["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", options=["No", "Yes"])
            partner = st.selectbox("Partner", options=["No", "Yes"])
            dependents = st.selectbox("Dependents", options=["No", "Yes"])
            tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
            tenure_group = "0-12 months" if tenure <= 12 else \
                           "13-24 months" if tenure <= 24 else \
                           "25-36 months" if tenure <= 36 else \
                           "37-48 months" if tenure <= 48 else \
                           "49-60 months" if tenure <= 60 else "61-72 months"
        
        # Service Information
        with col2:
            st.subheader("Service Information")
            phone_service = st.selectbox("Phone Service", options=["No", "Yes"])
            
            # Conditional fields based on phone service
            if phone_service == "Yes":
                multiple_lines = st.selectbox("Multiple Lines", options=["No", "Yes"])
            else:
                multiple_lines = "No phone service"
            
            internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
            
            # Conditional fields based on internet service
            if internet_service != "No":
                online_security = st.selectbox("Online Security", options=["No", "Yes"])
                online_backup = st.selectbox("Online Backup", options=["No", "Yes"])
                device_protection = st.selectbox("Device Protection", options=["No", "Yes"])
                tech_support = st.selectbox("Tech Support", options=["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes"])
            else:
                online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = "No internet service"
        
        # Account Information
        with col3:
            st.subheader("Account Information")
            contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", options=["No", "Yes"])
            payment_method = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.slider("Monthly Charges ($)", min_value=0, max_value=150, value=70)
            
            # Calculate derived features
            total_charges = monthly_charges * tenure if tenure > 0 else monthly_charges
            
            # Count total services
            services_count = sum([
                phone_service == "Yes",
                multiple_lines == "Yes",
                internet_service != "No",
                online_security == "Yes",
                online_backup == "Yes",
                device_protection == "Yes",
                tech_support == "Yes",
                streaming_tv == "Yes",
                streaming_movies == "Yes"
            ])
            
            # Calculate simple CLV
            if contract == "Month-to-month":
                expected_tenure = tenure + 6  # assume 6 more months
            elif contract == "One year":
                expected_tenure = tenure + 12  # assume at least 1 more year
            else:  # two year
                expected_tenure = tenure + 24  # assume at least 2 more years
            
            clv = monthly_charges * expected_tenure
            
            # Monthly charge category
            if monthly_charges < 35:
                monthly_charge_category = "Low"
            elif monthly_charges < 70:
                monthly_charge_category = "Medium"
            else:
                monthly_charge_category = "High"
                
            # Customer status based on tenure
            if tenure <= 12:
                customer_status = "New"
            elif tenure <= 36:
                customer_status = "Intermediate"
            elif tenure <= 60:
                customer_status = "Established"
            else:
                customer_status = "Loyal"
            
            # Average monthly charges
            avg_monthly_charges = total_charges / tenure if tenure > 0 else monthly_charges
        
        # Submit button
        submit_button = st.form_submit_button(label="Predict Churn Probability")
    
    # When form is submitted
    if submit_button:
        # Create a dictionary with customer data
        customer_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'TenureGroup': tenure_group,
            'TotalServices': services_count,
            'AvgMonthlyCharges': avg_monthly_charges,
            'MonthlyChargeCategory': monthly_charge_category,
            'Customer_Status': customer_status,
            'CLV': clv
        }
        
        # Convert to DataFrame for prediction
        customer_df = pd.DataFrame([customer_data])
        
        # Make prediction
        prediction = model.predict(customer_df)[0]
        prediction_proba = model.predict_proba(customer_df)[0, 1]  # Probability of class 1 (churn)
        
        # Get retention recommendations
        recommendations = get_retention_recommendations(prediction, prediction_proba, customer_data)
        
        # Display prediction results
        st.subheader("Prediction Result")
        
        # Display gauge chart for churn probability
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba * 100,
            title={'text': "Churn Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkgray"},
                'steps': [
                    {'range': [0, 30], 'color': "#3498db"},  # Low risk - blue
                    {'range': [30, 70], 'color': "#f39c12"},  # Medium risk - orange
                    {'range': [70, 100], 'color': "#e74c3c"}  # High risk - red
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction_proba * 100
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction interpretation
        st.subheader("Prediction Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Churn Prediction:** {'Yes' if prediction == 1 else 'No'}")
            st.markdown(f"**Churn Probability:** {prediction_proba*100:.2f}%")
            st.markdown(f"**Risk Level:** {recommendations['risk_level']}")
        
        with col2:
            st.subheader("Key Risk Factors")
            risk_factors = []
            
            if contract == "Month-to-month":
                risk_factors.append("Month-to-month contract")
            
            if internet_service == "Fiber optic":
                risk_factors.append("Fiber optic service")
                
            if tenure <= 12:
                risk_factors.append("New customer (under 12 months)")
                
            if payment_method == "Electronic check":
                risk_factors.append("Electronic check payment method")
                
            if services_count <= 3 and internet_service != "No":
                risk_factors.append("Few services relative to internet plan")
                
            if monthly_charges > 70:
                risk_factors.append("High monthly charges")
                
            if not risk_factors:
                st.write("No significant risk factors identified")
            else:
                for factor in risk_factors:
                    st.markdown(f"• {factor}")
        
        # Retention recommendations
        st.subheader("Retention Recommendations")
        for rec in recommendations['recommendations']:
            st.markdown(f"• {rec}")
        
        # Option to export results
        if st.button("Export Prediction Results"):
            export_data = {
                'prediction_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'churn_prediction': 'Yes' if prediction == 1 else 'No',
                'churn_probability': f"{prediction_proba*100:.2f}%",
                'risk_level': recommendations['risk_level'],
                'risk_factors': ", ".join(risk_factors) if risk_factors else "None",
                'recommendations': "\n".join(recommendations['recommendations'])
            }
            export_data.update(customer_data)
            
            # Create a DataFrame from the export data
            export_df = pd.DataFrame([export_data])
            
            # Convert to CSV
            csv = export_df.to_csv(index=False)
            
            # Create a download button
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="churn_prediction_result.csv",
                mime="text/csv"
            )

# Main function
def main():
    # Load data
    df = load_data()
    
    # Build model
    model, num_cols, cat_cols = build_model()
    
    # Render header
    render_header()
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Overview", "Model Performance", "Churn Prediction"])
    
    # Display the selected page
    if page == "Overview":
        overview_page(df)
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Churn Prediction":
        prediction_page(model, num_cols, cat_cols)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Telecom Customer Churn Prediction Dashboard**")
    st.sidebar.markdown("Developed with Streamlit and ML")
    st.sidebar.markdown(f"Last updated: {datetime.date.today().strftime('%Y-%m-%d')}")

# Run the app
if __name__ == "__main__":
    main()