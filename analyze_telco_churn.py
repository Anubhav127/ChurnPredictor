import pandas as pd
import numpy as np

print('Analyzing Telco Customer Churn dataset...')

# Load the dataset
df = pd.read_csv('/data/chats/iixf8/workspace/telco_customer_churn.csv')

# Basic dataset information
print('\nDataset shape:', df.shape)
print('\nFirst 5 rows:')
print(df.head())

# Check for missing values
print('\nMissing values:')
print(df.isnull().sum().sum(), 'total missing values')
print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().sum() > 0 else 'No columns with missing values')

# TotalCharges column check
print('\nTotalCharges data type:', df['TotalCharges'].dtype)
try:
    # If it's a string but contains numeric values
    if df['TotalCharges'].dtype == 'object':
        print('Converting TotalCharges to numeric...')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        print('Missing values after conversion:', df['TotalCharges'].isnull().sum())
except Exception as e:
    print(f'Error converting TotalCharges: {e}')

# Target variable analysis
print('\nTarget variable (Churn) distribution:')
churn_counts = df['Churn'].value_counts()
print(churn_counts)
churn_pcts = df['Churn'].value_counts(normalize=True)
print(f'Churn rate: {churn_pcts["Yes"]:.2%}')

# Key categorical features
print('\nContract type distribution:')
contract_counts = df['Contract'].value_counts()
print(contract_counts)

print('\nInternet service distribution:')
internet_counts = df['InternetService'].value_counts()
print(internet_counts)

print('\nPayment method distribution:')
payment_counts = df['PaymentMethod'].value_counts()
print(payment_counts)

# Churn rate by key features
print('\nChurn rate by contract type:')
for contract in df['Contract'].unique():
    churn_rate = df[df['Contract']==contract]['Churn'].value_counts(normalize=True).get('Yes', 0)
    print(f'{contract}: {churn_rate:.2%}')

print('\nChurn rate by internet service:')
for service in df['InternetService'].unique():
    churn_rate = df[df['InternetService']==service]['Churn'].value_counts(normalize=True).get('Yes', 0)
    print(f'{service}: {churn_rate:.2%}')

print('\nChurn rate by tenure groups:')
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                            labels=['0-12 mo', '13-24 mo', '25-36 mo', '37-48 mo', '49-60 mo', '61-72 mo'])
for group in df['tenure_group'].unique():
    churn_rate = df[df['tenure_group']==group]['Churn'].value_counts(normalize=True).get('Yes', 0)
    print(f'{group}: {churn_rate:.2%}')

# Summary statistics for numerical features
print('\nNumerical features statistics:')
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())

print('\nAnalysis complete.')
