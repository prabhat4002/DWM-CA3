# Shopping Trends Warehouse Predictor Report

## Objective
Develop a data warehousing application to predict the category of customer purchases based on demographic and transactional data, supporting retail decision-making.

## Dataset
- Source: Kaggle Customer Shopping Trends (~3,000 rows).
- Features: Age, Gender, Item Purchased, Purchase Amount, Location, Size, Color, Season, Review Rating, Subscription Status, Shipping Type, Discount Applied, Promo Code Used, Previous Purchases, Payment Method, Frequency of Purchases.
- Target: Category (e.g., Clothing, Footwear, Accessories, Outerwear).

## Methodology
- **Data Warehousing Context**: The app simulates a retail data warehouse by storing customer data (`shopping_trends_updated.csv`) and enabling query-like predictions via a form interface.
- **Model**: Random Forest Classifier to predict `Category` based on encoded features.
- **Interface**: Streamlit app with a sidebar form for manual input, displaying predictions and warehousing insights (e.g., average purchase amount by category).
- **Preprocessing**: Encoded categorical variables using LabelEncoder, dropped `Customer ID`.

## Results
- **Accuracy**: [Fill in after running, e.g., ~77% based on friend's result].
- **Key Features**: [Fill in top features, e.g., Item Purchased, Purchase Amount].
- **Insights**: 
  - Average purchase amount varies by category (e.g., Outerwear may be higher).
  - Seasonal trends show [e.g., Winter purchases dominate].

## Data Warehousing Relevance
- **Storage**: Dataset stored in `data/` as a flat file, mimicking a warehouse table.
- **Querying**: Form inputs act as queries to retrieve predictions.
- **Reporting**: Aggregate stats (e.g., purchase amount by category) support business intelligence.

## Challenges
- Encoding many categorical features required careful handling.
- Small dataset size (relative to real warehouses) limited some insights.

## Conclusion
The app demonstrates a practical data warehousing application for retail, combining machine learning with interactive querying. Future work could integrate a database (e.g., SQLite) for true warehousing.

**Submitted by**: [Your Name]