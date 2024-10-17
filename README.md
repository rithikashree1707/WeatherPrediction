# Weather Prediction Using Machine Learning

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
     
5. **Model Building:**
   - **Random Forest Classifier** is used to predict both high-impact precipitation and wind events.
   - **Logistic Regression** is applied to predict high-impact precipitation events.
   - **Decision Tree Classifier** is used to predict high-impact wind events.
   - Models are trained using 70% of the data, and predictions are made on the remaining 30%.

6. **Model Evaluation:**
   - Performance metrics such as accuracy, precision, recall, F1-score, and classification reports are used to evaluate each model.
   - Results of Random Forest, Logistic Regression, and Decision Tree models are compared to identify the best performing model.

7. **Model Comparison:**
   - A function is implemented to compare the performance of different models by plotting the metrics (accuracy, precision, recall, F1-score) for easy visual comparison.

### Required Libraries

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computation.
- **Matplotlib** and **Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning algorithms, model training, testing, and evaluation.
- **Warnings**: To suppress unnecessary warnings.

### Code Execution

1. **Data Loading:**
   - Load the dataset from a CSV file.
   - Preprocess the data by checking for missing or duplicate values and converting date columns.

2. **Exploratory Data Analysis (EDA):**
   - Inspect data types and distributions.
   - Visualize distributions and time-series plots for precipitation, temperature, and wind speed.

3. **Feature Engineering:**
   - Create new binary labels for high-impact events based on precipitation and wind thresholds.
   - Prepare features for the machine learning models.

4. **Model Training and Testing:**
   - Split the data into training and testing sets.
   - Train the models (Random Forest, Logistic Regression, Decision Tree) and make predictions.
   - Evaluate the models' performance using classification metrics.

5. **Model Comparison:**
   - Compare the models' performance using accuracy, precision, recall, and F1-score metrics.
   - Visualize the comparison to identify the best model.

### Key Variables

- **Precipitation**: Rainfall measured in mm across locations.
- **Temperature**: Mean temperature measured in °C.
- **Wind Speed**: Wind speed measured in m/s.
- **High-impact Precipitation/Wind**: Binary labels representing extreme weather conditions.

### Model Summary

- **Random Forest**: Used for both precipitation and wind prediction; provides robust performance by handling non-linearity in the data.
- **Logistic Regression**: Applied for precipitation prediction; works well for binary classification.
- **Decision Tree**: Used for wind prediction; good for capturing decision rules in data.

### Running the Code

1. Ensure the required libraries are installed: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
2. Load your dataset (`weather_prediction_dataset.csv`) into the `WeatherData` variable.
3. Run the code to preprocess, visualize, and model the weather data.
4. Evaluate and compare the performance of different machine learning models.
