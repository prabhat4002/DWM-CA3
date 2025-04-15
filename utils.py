import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, label_encoders=None, fit_encoders=True):
    """Preprocess dataset, encoding categorical columns securely."""
    df = df.copy()
    if "Customer ID" in df.columns:
        df = df.drop("Customer ID", axis=1)
    if "Invoice ID" in df.columns:
        df = df.drop("Invoice ID", axis=1)
    if "Item Purchased" in df.columns:
        df = df.drop("Item Purchased", axis=1, errors="ignore")
    
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    if label_encoders is None:
        label_encoders = {}
    
    for col in categorical_cols:
        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is None:
                raise ValueError(f"No encoder found for column {col}")
            df[col] = le.transform(df[col])
    
    return df, label_encoders

def get_warehouse_insights(df):
    """Compute aggregate stats for data warehousing."""
    insights = {}
    insights["avg_purchase_by_category"] = df.groupby("Category")["Purchase Amount (USD)"].mean().to_dict()
    insights["count_by_season"] = df["Season"].value_counts().to_dict()
    insights["count_by_category"] = df["Category"].value_counts().to_dict()
    return insights