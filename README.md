# Airline Delay Prediction Model

This project implements a machine learning model to predict airline arrival delays based on historical flight data. The model uses various factors such as weather conditions, carrier information, and airport statistics to forecast potential delays.

## Features

- Predicts arrival delay times in minutes
- Analyzes multiple delay causes (weather, security, late aircraft, etc.)
- Provides feature importance analysis
- Includes visualizations of model performance
- Interactive web interface for easy predictions

## Dataset

The model uses the `Airline_Delay_Cause_dataset.csv` file which was downloaded from the site of the Bureau of Transportation Statistics, and official website of the US Government. The link to the site is:
https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp?20=E

I used the filter that too data from January 2024 -  December 2024.
It contains the following key columns:
- `year`, `month`: Temporal information
- `carrier`, `carrier_name`: Airline information
- `airport`, `airport_name`: Airport information
- `arr_flights`: Number of arriving flights
- `arr_del15`: Number of flights delayed more than 15 minutes
- Various delay causes: `carrier_ct`, `weather_ct`, `nas_ct`, `security_ct`, `late_aircraft_ct`
- `arr_delay`: Target variable (arrival delay in minutes)

## Requirements

- Python 3.7+
- Required packages (see requirements.txt):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - streamlit

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Prediction Model Directly
```bash
python delay_prediction_model.py
```

The script will:
- Load and preprocess the data
- Train the Random Forest model
- Evaluate model performance
- Generate visualization plots:
  - `feature_importance.png`: Shows the most important factors affecting delays
  - `predicted_vs_actual.png`: Shows the accuracy of predictions

### Option 2: Run the Interactive Web App
```bash
streamlit run app.py
```

The web app provides:
- Interactive interface to select carrier and airport
- Real-time delay predictions
- Visualizations of:
  - Feature importance
  - Model performance
  - Predicted vs actual delays
- Key metrics:
  - R² Score
  - Mean Absolute Error (MAE)
  - Average and maximum delays in the dataset

## Model Performance

The model is evaluated using:
- R² Score: Measures how well the model explains the variation in delays
- Mean Absolute Error (MAE): Average error in minutes between predicted and actual delays

## Output

### Command Line Version
The script generates:
1. Console output with:
   - Data preprocessing information
   - Model performance metrics
2. Two visualization files:
   - `feature_importance.png`: Bar plot of top 15 most important features
   - `predicted_vs_actual.png`: Scatter plot comparing predicted vs actual delays

### Web App Version
The app provides:
1. Interactive interface with:
   - Carrier and airport selection
   - Real-time predictions
   - Model performance metrics
2. Visualizations:
   - Feature importance plot
   - Predicted vs actual delay plot
3. Dataset statistics in the sidebar

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. 
