"""Data utility functions for Loan Prediction project"""
import pandas as pd
import numpy as np




def load_csv(path: str) -> pd.DataFrame:
"""Load CSV with common encodings"""
return pd.read_csv(path)




def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
df = df.copy()
# drop Loan_ID
if 'Loan_ID' in df.columns:
df = df.drop(columns=['Loan_ID'])


# fill categorical with mode
cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
for c in cat_cols:
if c in df.columns:
df[c] = df[c].fillna(df[c].mode().iloc[0])


# fill numerics
num_cols = ['LoanAmount', 'Loan_Amount_Term']
for c in num_cols:
if c in df.columns:
df[c] = pd.to_numeric(df[c], errors='coerce')
df[c] = df[c].fillna(df[c].mean())


# replace Dependents '3+'
if 'Dependents' in df.columns:
df['Dependents'] = df['Dependents'].astype(str).str.replace('+', '', regex=False)
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0).astype(int)


return df
