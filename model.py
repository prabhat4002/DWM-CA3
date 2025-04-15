import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import preprocess_data

def train_model(data_path, model_type="RandomForest", target="Category"):
    """Train specified model with error handling."""
    try:
        df = pd.read_csv(data_path)
        df_processed, label_encoders = preprocess_data(df, fit_encoders=True)
        
        X = df_processed.drop(target, axis=1, errors="ignore")
        y = df_processed[target]
        
        if model_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
            plt.title('Top 10 Feature Importance (Random Forest)')
            feature_importance_path = "images/feature_importance.png"
            plt.savefig(feature_importance_path)
            plt.close()
            
            return model, label_encoders, feature_importance_path, None
        
        elif model_type == "NaiveBayes":
            model = GaussianNB()
            model.fit(X, y)
            
            return model, label_encoders, None, None
    
    except Exception as e:
        raise Exception(f"Model training failed: {str(e)}")

def predict_category(model, label_encoders, input_data, category_encoder):
    """Predict category for a single input with error handling."""
    try:
        input_df = pd.DataFrame([input_data])
        input_processed, _ = preprocess_data(input_df, label_encoders, fit_encoders=False)
        
        prediction = model.predict(input_processed)[0]
        probabilities = model.predict_proba(input_processed)[0]
        
        category = category_encoder.inverse_transform([prediction])[0]
        prob_dict = {category_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}
        
        return category, prob_dict
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")