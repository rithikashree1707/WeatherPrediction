# Weather Prediction Using Machine Learning

## Project Structure

1. **Data Loading and Exploration**:
   - The historical weather dataset is loaded using **Pandas**.
   - Various features such as temperature, humidity, pressure, and wind speed across different locations are explored.
   - Exploratory Data Analysis (EDA) involves descriptive statistics, checking for missing values, and duplicate rows.

2. **Data Visualization**:
   - Visualize time-series trends for temperature, precipitation, and wind speed.
   - Visualize distributions and correlations between weather variables.
   - Use histograms, time-series plots, and correlation matrices for better insights into the data.

3. **Feature Engineering**:
   - Define **high-impact weather events** by identifying the top 10% of extreme precipitation and wind speed values.
   - Create binary labels (1 = high-impact event, 0 = not high-impact) for classification.
   - Select relevant features like temperature, humidity, and wind speed for modeling.

4. **Machine Learning Models**:
   - **Random Forest Classifier**: Predicts high-impact weather events for precipitation and wind.
   - **Logistic Regression**: Another classifier for predicting high-impact precipitation.
   - **Decision Tree Classifier**: Used to classify high-impact wind events.

5. **Model Evaluation**:
   - Use metrics such as **accuracy**, **precision**, **recall**, and **F1-score** to evaluate the performance of classification models.
   - Visualize model performance using **model comparison plots**.

6. **Weather Prediction (Regression)**:
   - Implement **Random Forest Regressor** to predict continuous weather variables, such as mean temperature.
   - Evaluate the regression model using **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **Root Mean Squared Error (RMSE)**.
   - Visualize actual vs. predicted temperature values using scatter plots and time-series plots.

7. **Forecasting**:
   - Forecast future weather conditions (e.g., temperature) using the trained regression model.
   - Predict the temperature for the next 7 days based on manually created future data inputs.
   - Visualize the forecasted temperature over time.

---

## Project Dependencies

- **Python Libraries**:
  - `pandas`: For data loading and manipulation.
  - `numpy`: For numerical operations.
  - `matplotlib`, `seaborn`: For data visualization.
  - `sklearn`: For machine learning models, including classifiers and regressors.

---

## How to Run the Project

### Step 1: Install Required Libraries

Before running the code, ensure that all required libraries are installed. Use the following commands:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Step 2: Load the Weather Dataset

Ensure that your weather dataset (e.g., `weather_prediction_dataset.csv`) is correctly referenced in the script.

```python
WeatherData = pd.read_csv("path/to/weather_prediction_dataset.csv")
```

### Step 3: Perform Data Exploration and Visualization

The script includes various data exploration and visualization steps, such as plotting time-series data for temperature, precipitation, and wind speed. Use these to gain insights into the weather trends over time.

```python
# Example: Time-series plot for Basel temperature
plt.plot(WeatherData['DATE'], WeatherData['BASEL_temp_mean'])
plt.show()
```

### Step 4: Run Classification Models

The project includes the implementation of **Random Forest**, **Logistic Regression**, and **Decision Tree Classifiers** to predict high-impact weather events like extreme precipitation or wind. The models are trained on 70% of the data and tested on the remaining 30%.

```python
# Example: Train and test Random Forest for high-impact precipitation prediction
RandomForestPrecipitation.fit(X_trainPrecipitation, y_trainPrecipitation)
RandomForestPrecipitationPrediction = RandomForestPrecipitation.predict(X_testPrecipitation)
```

### Step 5: Model Evaluation and Comparison

After training the models, evaluate their performance using metrics like accuracy, precision, recall, and F1-score. Visualize the model comparison using the provided comparison plotting function.

```python
# Example: Model comparison plot
ModelComparisonPlottingFunction([RandomForestPrecipitation, LogisticRegressionPrecipitation], X_testPrecipitation, y_testPrecipitation, ["Random Forest", "Logistic Regression"])
```

### Step 6: Weather Forecasting (Temperature Prediction)

The **Random Forest Regressor** is used to predict temperature. The model is trained on historical data and used to predict future temperature trends. The future temperature forecast is plotted for the next 7 days based on custom inputs.

```python
# Example: Predict future temperature using Random Forest Regressor
FutureTemperaturePredictions = RandomForestRegressor.predict(FutureData)
```

### Step 7: Visualization of Forecasts

Visualize the future temperature forecast using line plots.

```python
# Example: Forecast visualization for the next 7 days
plt.plot(FutureDates, FutureTemperaturePredictions)
plt.show()
```
