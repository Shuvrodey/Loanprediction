"""Modeling utilities: pipelines, training and evaluation"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix




def build_preprocessor(X: pd.DataFrame):
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()


num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])


ct = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')
return ct




def evaluate_model(model, X_test, y_test):
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None
metrics = {
'accuracy': accuracy_score(y_test, y_pred),
'precision': precision_score(y_test, y_pred),
'recall': recall_score(y_test, y_pred),
'f1': f1_score(y_test, y_pred),
}
if y_prob is not None:
metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
return metrics, confusion_matrix(y_test, y_pred)
