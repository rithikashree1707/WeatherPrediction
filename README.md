# Weather Prediction Using Machine Learning

## Research Topic: Machine Learning and Numerical Prediction Models for High-Impact Weather Prediction

This project is aimed at building a machine learning-based prediction model for high-impact weather events, focusing specifically on extreme precipitation and wind events. The dataset used contains meteorological measurements from different weather stations (e.g., Basel, Tours, DE BILT), and the goal is to predict these extreme events based on various weather parameters.

### Project Structure

1. **Data Loading and Preprocessing**
   - Load the weather dataset `weather_prediction_dataset.csv`.
   - Inspect the dataset for structure, missing values, and unique values in each column.
   - Convert relevant columns (e.g., dates) to appropriate formats (such as `datetime`).
   - Check for duplicate rows and remove them if necessary.

2. **Exploratory Data Analysis (EDA)**
   - Generate descriptive statistics to understand the data distribution.
   - Visualize key weather variables such as temperature, precipitation, and wind speed over time using histograms and time-series plots.
   - Analyze the correlation between various weather features to identify relationships that may impact weather event prediction.

3. **High-Impact Weather Event Definition**
   - Use thresholds for defining extreme events:
     - **High-impact precipitation**: Defined by the 90th percentile of the `BASEL_precipitation` values.
     - **High-impact wind speed**: Defined by the 90th percentile of the `DE_BILT_wind_speed` values.
   - Create binary labels for these high-impact events (1 for extreme events, 0 otherwise).

4. **Data Splitting**
   - Select relevant features, including temperature, humidity, pressure, and wind speed, from different weather stations.
   - Set up target variables for both high-impact precipitation and wind events.
   - Split the data into training and testing sets (70% training, 30% testing) using `train_test_split` for both high-impact precipitation and wind predictions.
