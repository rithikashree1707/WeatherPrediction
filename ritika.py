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

# Importing accuracy metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Ignoring warnings
import warnings
warnings.filterwarnings("ignore")

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

