# Air Pollution

This project is a machine learning solution designed to analyze and predict pollution levels based on a given dataset. The model is trained using supervised learning techniques and aims to provide accurate predictions for air quality monitoring.

## Project Files

- **`model.joblib`**: Pre-trained machine learning model for pollution prediction.
- **`reduced_pollution_dataset.csv`**: Processed dataset used for training and testing the model.
- **`Untitled.ipynb`**: Jupyter Notebook containing the implementation of the machine learning pipeline, including data preprocessing, model training, evaluation, and visualization.

## Usage

1) Open the Jupyter Notebook:

   ```bash
   jupyter notebook Untitled.ipynb
   ```

2) Run the notebook cells to:

   - Load the dataset
   - Preprocess the data
   - Train the machine learning model
   - Evaluate the performance
   - Make predictions on new data

3) To use the pre-trained model for predictions:

   ```python
   import joblib
   import pandas as pd

   model = joblib.load('model.joblib')
   data = pd.read_csv('reduced_pollution_dataset.csv')
   predictions = model.predict(data)
   print(predictions)
   ```

## Dataset

The `reduced_pollution_dataset.csv` contains environmental data such as:

- Temperature
- Humidity
- Air pollutants levels (PM2.5, PM10, NO2, SO2, etc.)
- Wind speed

## Model

The machine learning model was trained using algorithms such as:

- Decision Trees

## Evaluation Metrics

The model performance was evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R^2) score

##

