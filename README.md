# Shopping Trends Warehouse Predictor

Run the app: https://dwm-ca3-group12.streamlit.app/

A Streamlit app for a data warehousing course, predicting purchase categories using a Random Forest model with a form-based interface.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place `shopping_trends_updated.csv` in `data/`
3. Run: `streamlit run app.py`

## Structure
- `data/`: Dataset storage.
- `images/`: Generated plots.
- `app.py`: Streamlit app with form.
- `model.py`: Model training/prediction.
- `utils.py`: Preprocessing and insights.
- `requirements.txt`: Dependencies.
- `report_template.md`: Submission report template.

## Features
- Form to input customer details.
- Predicts purchase category with probabilities.
- Shows warehouse insights (e.g., avg purchase by category).
- Displays feature importance.
