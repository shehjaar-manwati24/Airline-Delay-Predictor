import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Airline Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

# Cache data loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    # Load data
    df = pd.read_csv('Airline_Delay_Cause.csv')
    
    # Drop rows with missing values in target variable
    df = df.dropna(subset=['arr_delay'])
    
    # Convert categorical variables
    categorical_cols = ['carrier', 'carrier_name', 'airport', 'airport_name']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    # Create ratio features
    df['delay_ratio'] = df['arr_del15'] / df['arr_flights']
    df['cancellation_ratio'] = df['arr_cancelled'] / df['arr_flights']
    df['diversion_ratio'] = df['arr_diverted'] / df['arr_flights']
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['carrier', 'airport'])
    
    # Select features
    feature_cols = [
        'year', 'month', 'arr_flights', 'delay_ratio', 'cancellation_ratio', 
        'diversion_ratio', 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 
        'late_aircraft_ct'
    ] + [col for col in df_encoded.columns if col.startswith(('carrier_', 'airport_'))]
    
    X = df_encoded[feature_cols]

    # Force all columns to be numeric and handle any leftover non-numeric data
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    # Debug print (optional — remove after testing)
    print("Non-numeric columns in X:", X.select_dtypes(include=['object', 'category']).columns)


    y = df_encoded['arr_delay']
    
    return df, X, y, feature_cols

# Cache model training
@st.cache_data
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Main app
def main():
    st.title("✈️ Airline Delay Predictor")
    st.markdown("""
    This app predicts arrival delays for flights based on historical data. 
    Select a carrier and airport to see the predicted average delay.
    """)
    
    # Load and preprocess data
    df, X, y, feature_cols = load_and_preprocess_data()
    
    # Train model
    model = train_model(X, y)
    
    # Sidebar for user inputs
    st.sidebar.header("Flight Details")
    
    # Get unique carriers and airports
    carriers = sorted(df['carrier'].unique())
    airports = sorted(df['airport'].unique())
    
        # Get unique carriers and airports
    carriers = sorted(df['carrier'].unique())
    
    # Create airport options with both code and name
    airport_options = df[['airport', 'airport_name']].drop_duplicates()
    airport_options['display'] = airport_options['airport'].astype(str) + ' - ' + airport_options['airport_name'].astype(str)
    airport_options = airport_options.sort_values('airport')
    
    # User inputs
    selected_carrier = st.sidebar.selectbox("Select Carrier", carriers)
    selected_airport_display = st.sidebar.selectbox(
        "Select Airport", 
        airport_options['display']
    )
    
    # Extract airport code from the selected display option
    selected_airport = selected_airport_display.split(' - ')[0]
    
    # Create feature vector for prediction
    def create_feature_vector(carrier, airport):
        # Create a row with all features set to their means
        feature_vector = X.mean().copy()
        
        # Set carrier and airport one-hot encoding
        carrier_col = f'carrier_{carrier}'
        airport_col = f'airport_{airport}'
        
        if carrier_col in feature_vector.index:
            feature_vector[carrier_col] = 1
        if airport_col in feature_vector.index:
            feature_vector[airport_col] = 1
            
        return feature_vector
    
    
    # Make prediction
    feature_vector = create_feature_vector(selected_carrier, selected_airport)
    predicted_delay = model.predict([feature_vector])[0]
    
    # Display prediction
    st.header("Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Predicted Average Delay",
            value=f"{predicted_delay:.1f} minutes",
            delta=None
        )
    
    # Model evaluation metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    with col2:
        st.metric(
            label="Model R² Score",
            value=f"{r2:.3f}",
            delta=None
        )
        st.metric(
            label="Mean Absolute Error",
            value=f"{mae:.1f} minutes",
            delta=None
        )
    
    # Feature importance plot
    st.header("Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    st.pyplot(plt)
    
    # Predicted vs Actual plot
    st.header("Model Performance")
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Delay (minutes)')
    plt.ylabel('Predicted Delay (minutes)')
    plt.title('Actual vs Predicted Delay')
    plt.tight_layout()
    st.pyplot(plt)
    
    # Data description
    st.sidebar.header("About the Data")
    st.sidebar.markdown(f"""
    - Number of flights: {len(df):,}
    - Average delay: {df['arr_delay'].mean():.1f} minutes
    - Maximum delay: {df['arr_delay'].max():.1f} minutes
    """)

if __name__ == "__main__":
    main() 