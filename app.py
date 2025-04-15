import streamlit as st
import pandas as pd
import os
from model import train_model, predict_category
from utils import get_warehouse_insights

# Set page config
st.set_page_config(page_title="DWM CA3", layout="wide")

# Create images directory
if not os.path.exists("images"):
    os.makedirs("images")

# Sidebar: Model selection
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox("Choose Model", ["RandomForest", "NaiveBayes"])

# Title and instructions
st.title("🛍️ DWM CA3")
st.info("Select a model in the sidebar, enter customer details below, and predict the purchase category. Detailed results are available after submission.")

# Load and train model
data_path = "data/shopping_trends_updated.csv"
if not os.path.exists(data_path):
    st.error("Please place 'shopping_trends_updated.csv' in the 'data/' folder.")
    st.stop()

with st.spinner("Training model..."):
    try:
        model, label_encoders, feature_importance_path, confusion_matrix_path = train_model(data_path, model_type)
        df = pd.read_csv(data_path)
        insights = get_warehouse_insights(df)
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        st.stop()

# Main page: Input Form
st.header("Enter Customer Details")
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        purchase_amount = st.number_input("Purchase Amount (USD)", min_value=0, value=50)
        location = st.selectbox("Location", df["Location"].unique())
        size = st.selectbox("Size", ["S", "M", "L", "XL"])
        
    with col2:
        color = st.selectbox("Color", df["Color"].unique())
        season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
        review_rating = st.number_input("Review Rating", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        subscription_status = st.selectbox("Subscription Status", ["Yes", "No"])
        shipping_type = st.selectbox("Shipping Type", df["Shipping Type"].unique())
        
    with col3:
        discount_applied = st.selectbox("Discount Applied", ["Yes", "No"])
        promo_code_used = st.selectbox("Promo Code Used", ["Yes", "No"])
        previous_purchases = st.number_input("Previous Purchases", min_value=0, value=10)
        payment_method = st.selectbox("Payment Method", df["Payment Method"].unique())
        frequency_of_purchases = st.selectbox("Frequency of Purchases", df["Frequency of Purchases"].unique())
    
    submit_button = st.form_submit_button("Predict Category")

# Prediction Result
if submit_button:
    input_data = {
        "Age": age,
        "Gender": gender,
        "Purchase Amount (USD)": purchase_amount,
        "Location": location,
        "Size": size,
        "Color": color,
        "Season": season,
        "Review Rating": review_rating,
        "Subscription Status": subscription_status,
        "Shipping Type": shipping_type,
        "Discount Applied": discount_applied,
        "Promo Code Used": promo_code_used,
        "Previous Purchases": previous_purchases,
        "Payment Method": payment_method,
        "Frequency of Purchases": frequency_of_purchases
    }
    
    st.header("Prediction Result")
    with st.spinner("Making prediction..."):
        try:
            category, probabilities = predict_category(
                model, label_encoders, input_data, label_encoders["Category"]
            )
            
            st.success(f"🎉 Predicted Category: **{category}**")
            
            # Expandable detailed results
            with st.expander("View Detailed Results"):
                st.subheader("Prediction Confidence")
                st.write({k: f"{v:.2%}" for k, v in probabilities.items()})
                
                st.subheader("Similar Customers (Warehouse Query)")
                similar_df = df[
                    (df["Gender"] == gender) &
                    (df["Season"] == season) &
                    (df["Category"] == category)
                ].head(5)[["Age", "Purchase Amount (USD)", "Location", "Category"]]
                st.write("Top 5 matching records from the warehouse:")
                st.dataframe(similar_df)
                
                st.subheader("Data Warehouse Insights")
                st.write("**Average Purchase Amount by Category**")
                st.write({k: f"${v:.2f}" for k, v in insights["avg_purchase_by_category"].items()})
                st.write("**Purchase Count by Season**")
                st.write(insights["count_by_season"])
                st.write("**Purchase Count by Category**")
                st.write(insights["count_by_category"])
                
                st.subheader("Model Insights")
                if model_type == "RandomForest" and feature_importance_path:
                    st.image(feature_importance_path, use_column_width=True, caption="Feature Importance (Random Forest)")
                elif model_type == "NaiveBayes" and confusion_matrix_path:
                    st.image(confusion_matrix_path, use_column_width=True, caption="Confusion Matrix (Gaussian Naive Bayes)")
            
            st.download_button(
                "Download Prediction",
                f"Predicted Category: {category}\nProbabilities: {probabilities}",
                file_name="prediction.txt"
            )
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Footer
