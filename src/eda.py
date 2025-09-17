"""Exploratory Data Analysis functions"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




def missing_barplot(df: pd.DataFrame, figsize=(10,6)):
missing_counts = df.isnull().sum()
plt.figure(figsize=figsize)
sns.barplot(x=missing_counts.values, y=missing_counts.index)
plt.title('Missing Values per Column')
plt.xlabel('Count of Missing Values')
plt.ylabel('Columns')
plt.tight_layout()
return plt




def plot_loan_status(df: pd.DataFrame):
plt.figure(figsize=(6,5))
sns.countplot(x='Loan_Status', data=df)
plt.title('Distribution of Loan Status')
return plt
