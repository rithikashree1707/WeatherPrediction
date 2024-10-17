# -*- coding: utf-8 -*-

# Importing necessary libraries for

# Data analysis and manipulation
import numpy as np
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Training testing split
from sklearn.model_selection import train_test_split

# Importing Classifier Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Importing Regression model
from sklearn.ensemble import RandomForestRegressor

# Importing accuracy metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, mean_absolute_error, mean_squared_error

# Ignoring warnings
import warnings
warnings.filterwarnings("ignore")

"""## Exploratory Data Analysis (EDA)"""

# Data loading
WeatherData = pd.read_csv("/content/weather_prediction_dataset.csv")

# Checking the information about the data
WeatherData.info()

# Printing the first few data values
WeatherData.head()

# Printing the last few data values
WeatherData.tail()

# Checking the descriptive stats of the data
WeatherData.describe()

# Checking for missing values
WeatherData.isnull().sum()

# Data Type Distribution
WeatherData.dtypes.value_counts()

# Unique Values for Features
for column in WeatherData.columns:
  print(f"\nUnique values in {column}: {WeatherData[column].nunique()}")

# Checking for duplicate rows
DuplicateRows = WeatherData[WeatherData.duplicated()]
print(f"\nNumber of duplicate rows: {len(DuplicateRows)}")

"""## Data Visualisation"""

# Converting the 'DATE' column to a datetime format
WeatherData['DATE'] = pd.to_datetime(WeatherData['DATE'], format='%Y%m%d')

# Selecting the columns for distribution of some key variables
VariablesOfInterest = ['BASEL_precipitation', 'BASEL_temp_mean', 'TOURS_precipitation',
                       'TOURS_temp_mean', 'TOURS_wind_speed']

# Plotting distribution of key variables to identify thresholds for high-impact events
WeatherData[VariablesOfInterest].hist(bins=50, figsize=(15,10))
plt.tight_layout()
plt.show()

# Plotting time-series for Basel Temperature over time
plt.figure(figsize=(12, 6))
plt.plot(WeatherData['DATE'], WeatherData['BASEL_temp_mean'], label='Basel Temp Mean')
plt.plot(WeatherData['DATE'], WeatherData['TOURS_temp_mean'], label='Tours Temp Mean')
plt.title('Temperature Trends Over Time')
plt.ylabel('Mean Temperature (°C)')
plt.legend()
plt.show()

# Plotting time-series for Precipitation Trends
plt.figure(figsize=(12, 6))
plt.plot(WeatherData['DATE'], WeatherData['BASEL_precipitation'], label='Basel Precipitation')
plt.plot(WeatherData['DATE'], WeatherData['TOURS_precipitation'], label='Tours Precipitation')
plt.title('Precipitation Trends Over Time')
plt.ylabel('Precipitation (mm)')
plt.legend()
plt.show()

# Plotting time-series for Wind Speed Trends
plt.figure(figsize=(12, 6))
plt.plot(WeatherData['DATE'], WeatherData['DE_BILT_wind_speed'], label='DE BILT Wind Speed')
plt.plot(WeatherData['DATE'], WeatherData['TOURS_wind_speed'], label='Tours Wind Speed')
plt.title('Wind Speed Trends Over Time')
plt.ylabel('Wind Speed (m/s)')
plt.legend()
plt.show()

# Caclculating Correlation using variables related to precipitation, temperature, and wind speed across different locations.
CorrelationMatrix = WeatherData[['BASEL_precipitation', 'BASEL_temp_mean', 'DE_BILT_wind_speed',
                                   'TOURS_precipitation', 'TOURS_temp_mean', 'TOURS_wind_speed']].corr()

# Plotting the correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(CorrelationMatrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Key Weather Variables')
plt.show()

"""## Data Processing for Modelling"""

# Defining thresholds for high-impact precipitation and wind speed
PrecipitationThreshold = WeatherData['BASEL_precipitation'].quantile(0.90)  # Top 10% for precipitation
WindSpeedThreshold = WeatherData['DE_BILT_wind_speed'].quantile(0.90)  # Top 10% for wind speed

# Creating binary labels for high-impact events (1 = high-impact, 0 = not high-impact)
WeatherData['high_impact_precipitation'] = (WeatherData['BASEL_precipitation'] >= PrecipitationThreshold).astype(int)
WeatherData['high_impact_wind'] = (WeatherData['DE_BILT_wind_speed'] >= WindSpeedThreshold).astype(int)

# Selecting features for the model
features = ['BASEL_temp_mean', 'BASEL_humidity', 'BASEL_pressure', 'TOURS_temp_mean',
            'TOURS_humidity', 'TOURS_pressure', 'DE_BILT_wind_speed', 'TOURS_wind_speed']

# Setting Target variables
TargetPrecipitation = 'high_impact_precipitation'
TargetWind = 'high_impact_wind'

# Splitting data into train and test sets
X = WeatherData[features]
yPrecipitation = WeatherData[TargetPrecipitation]
yWind = WeatherData[TargetWind]

# Separately splitting for both high-impact precipitation and wind events
X_trainPrecipitation, X_testPrecipitation, y_trainPrecipitation, y_testPrecipitation = train_test_split(X, yPrecipitation, test_size=0.3, random_state=42)
X_trainWind, X_testWind, y_trainWind, y_testWind = train_test_split(X, yWind, test_size=0.3, random_state=42)

"""## Implementing Modelling"""

# Implementing Random Forest Model for high-impact precipitation prediction
RandomForestPrecipitation = RandomForestClassifier(random_state=42)
RandomForestPrecipitation.fit(X_trainPrecipitation, y_trainPrecipitation)

# Making predictions for the Random Forest Precipitation Model using testing data
RandomForestPrecipitationPrediction = RandomForestPrecipitation.predict(X_testPrecipitation)

# Implementing Random Forest Model for high-impact wind prediction
RandomForestModelWind = RandomForestClassifier(random_state=42)
RandomForestModelWind.fit(X_trainWind, y_trainWind)

# Making predictions for the Random Forest Wind Model using testing data
RandomForestModelWindPredictions = RandomForestModelWind.predict(X_testWind)

# Calculating the performance of both models
RandomForestPrecipitationReport = classification_report(y_testPrecipitation, RandomForestPrecipitationPrediction)
RandomForestModelWindReport = classification_report(y_testWind, RandomForestModelWindPredictions)

# Printing the performance of Random Forest Model for Precipitation
print("Random Forest Model for High-Impact Precipitation Prediction Performance:\n", RandomForestPrecipitationReport)

# Printing the performance of Random Forest Model for Wind
print("Random Forest Model for High-Impact Wind Prediction Performance:\n", RandomForestModelWindReport)

# Implementing Logistic Regression Model for high-impact precipitation prediction
LogisticRegressionPrecipitation = LogisticRegression(random_state=42, max_iter=1000)
LogisticRegressionPrecipitation.fit(X_trainPrecipitation, y_trainPrecipitation)

# Making predictions for the Logistic Regression Precipitation Model using testing data
LogisticRegressionPrecipitationPrediction = LogisticRegressionPrecipitation.predict(X_testPrecipitation)

# Calculating the performance of the model
LogisticRegressionPrecipitationReport = classification_report(y_testPrecipitation, LogisticRegressionPrecipitationPrediction)

# Printing the performance of Logistic Regression Model for Precipitation
print("Logistic Regression Model for High-Impact Precipitation Prediction Performance:\n", LogisticRegressionPrecipitationReport)

# Implementing Decision Tree Model for high-impact wind prediction
DecisionTreeModelWind = DecisionTreeClassifier(random_state=42)
DecisionTreeModelWind.fit(X_trainWind, y_trainWind)

# Making predictions for the Decision Tree Wind Model using testing data
DecisionTreeModelWindPredictions = DecisionTreeModelWind.predict(X_testWind)

# Calculating the performance of the model
DecisionTreeModelWindReport = classification_report(y_testWind, DecisionTreeModelWindPredictions)

# Printing the performance of Decision Tree Model for Wind
print("Decision Tree Model for High-Impact Wind Prediction Performance:\n", DecisionTreeModelWindReport)

"""## Model Comparison"""

# Defining a function to plot the model comparison
def ModelComparisonPlottingFunction(models, X_test, y_test, ModelNames):
  metrics = {
      'Accuracy': accuracy_score,
      'Precision': precision_score,
      'Recall': recall_score,
      'F1-Score': f1_score,
  }

  results = {}
  for model, name in zip(models, ModelNames):
    y_pred = model.predict(X_test)
    results[name] = {}
    for MetricName, MetricFunction in metrics.items():
      try:
        results[name][MetricName] = MetricFunction(y_test, y_pred)
      except ValueError:
        results[name][MetricName] = np.nan  # Handling cases where a metric can't be computed

  DataframeResults = pd.DataFrame(results).transpose()

  # Plotting the comparison
  plt.figure(figsize=(12, 6))
  for MetricName in metrics:
    plt.plot(DataframeResults.index, DataframeResults[MetricName], label=MetricName, marker='o')
  plt.title('Model Comparison')
  plt.xlabel('Model')
  plt.ylabel('Score')
  plt.xticks(rotation=45, ha='right')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()

# Applying the function on the model
models = [RandomForestPrecipitation, LogisticRegressionPrecipitation, RandomForestModelWind, DecisionTreeModelWind]
X_tests = [X_testPrecipitation, X_testPrecipitation, X_testWind, X_testWind]
y_tests = [y_testPrecipitation, y_testPrecipitation, y_testWind, y_testWind]
ModelNames = ['Random Forest (Precipitation)', 'Logistic Regression (Precipitation)', 'Random Forest (Wind)', 'Decision Tree (Wind)']

# Calling the function for each target variable
for X_test, y_test, model_name in zip(X_tests, y_tests, ModelNames):
  ModelComparisonPlottingFunction([models[i] for i in range(len(models))], X_test, y_test, ModelNames)

"""## Weather Prediction"""

# Feature selection for prediction (using example temperature and wind-related features)
FeaturesForPrediction = ['BASEL_humidity', 'BASEL_pressure', 'TOURS_temp_mean', 'TOURS_humidity',
                           'TOURS_pressure', 'DE_BILT_wind_speed', 'TOURS_wind_speed']

# Target variable for prediction: let's predict the 'BASEL_temp_mean'
TargetForPrediction = 'BASEL_temp_mean'

# Splitting data into features (X) and target (y)
X_prediction = WeatherData[FeaturesForPrediction]
y_prediction = WeatherData[TargetForPrediction]

# Splitting data into training and testing sets (70% train, 30% test)
XTrainPrediction, XTestPrediction, yTrainPrediction, yTestPrediction = train_test_split(X_prediction, y_prediction, test_size=0.3, random_state=42)

# Initialising the Random Forest Regressor
RandomForestRegressor = RandomForestRegressor(random_state=42)

# Training the Random Forest Regressor
RandomForestRegressor.fit(XTrainPrediction, yTrainPrediction)

# Making predictions using the test data
RandomForestRegressorPredictions = RandomForestRegressor.predict(XTestPrediction)

# Calculating the Random Forest Regressor metrics
mae = mean_absolute_error(yTestPrediction, RandomForestRegressorPredictions)
mse = mean_squared_error(yTestPrediction, RandomForestRegressorPredictions)
rmse = mean_squared_error(yTestPrediction, RandomForestRegressorPredictions, squared=False)

# Printing evaluation metrics
print(f"Random Forest Regressor for Temperature Prediction")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plotting predicted vs actual temperatures
plt.figure(figsize=(10, 6))
plt.scatter(yTestPrediction, RandomForestRegressorPredictions, color='blue', alpha=0.5)
plt.plot([yTestPrediction.min(), yTestPrediction.max()], [yTestPrediction.min(), yTestPrediction.max()], 'k--', lw=3)
plt.xlabel('Actual Temperatures')
plt.ylabel('Predicted Temperatures')
plt.title('Random Forest Temperature Prediction: Actual vs Predicted')
plt.show()

# Plotting a time-series graph of actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(yTestPrediction.values[:100], label='Actual Temperature', marker='o')
plt.plot(RandomForestRegressorPredictions[:100], label='Predicted Temperature', marker='x')
plt.title('Actual vs Predicted Temperature Over Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()

"""## Forecasting"""

# Creating a dataframe for Forecasting
FutureDays = 7
FutureData = pd.DataFrame({
    'BASEL_humidity': [70, 72, 75, 78, 80, 82, 85],
    'BASEL_pressure': [1012, 1010, 1008, 1006, 1004, 1002, 1000],
    'TOURS_temp_mean': [20, 21, 22, 23, 24, 25, 26],
    'TOURS_humidity': [60, 62, 64, 66, 68, 70, 72],
    'TOURS_pressure': [1015, 1013, 1011, 1009, 1007, 1005, 1003],
    'DE_BILT_wind_speed': [5, 6, 7, 8, 9, 10, 11],
    'TOURS_wind_speed': [3, 4, 5, 6, 7, 8, 9]
})

# Using Random Forest Regressor to predict future temperature
FutureTemperaturePredictions = RandomForestRegressor.predict(FutureData)

# Creating a date range for the next 7 days
today = pd.Timestamp.today()
FutureDates = [today + pd.DateOffset(days=i) for i in range(1, FutureDays + 1)]

# Plotting the future temperature predictions
plt.figure(figsize=(12, 6))
plt.plot(FutureDates, FutureTemperaturePredictions, marker='o', label='Predicted Temperature')
plt.title('Temperature Forecast for the Next 7 Days')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# Printing the future temperature predictions with dates
for date, temp in zip(FutureDates, FutureTemperaturePredictions):
  print(f"Date: {date.date()} - Predicted Temperature: {temp:.2f} °C")