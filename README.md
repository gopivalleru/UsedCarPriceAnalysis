# What Drives the Price of a Car?

This project investigates a dataset of used cars to pinpoint factors affecting car prices and offers actionable insights for used car dealerships on consumer preferences.

[Data File](data/vehicles.csv)

## Prerequisites

### Tools and Environment

- **Jupyter Notebook**: Preferably via Anaconda-Navigator or any IDE supporting Jupyter Notebooks.
- **Python Version**: 3.11.5

### Essential Libraries

- matplotlib 3.7.2
- seaborn 0.12.2
- pandas 2.0.3
- plotly 5.9.0

## Exploratory Data Analysis

The analysis, complete with visualizations and detailed commentary, is thoroughly documented in the [Jupyter Notebook](./prompt_II.ipynb).

## Data Overview

The dataset initially contains 426,879 entries across 18 columns. Key columns show varying levels of missing data:

- `year`: 0.282%
- `manufacturer`: 4.134%
- `model`: 1.236%
- `condition`: 40.785%
- `cylinders`:      41.622%
- `fuel`:            0.706%
- `odometer`:        1.031%
- `title_status`:    1.931%
- `transmission`:    0.599%
- `VIN`:            37.725%
- `drive`:          30.586%
- `size`:           71.767%
- `type`:           21.753%
- `paint_color`:    30.501%

## Data Cleaning

### Addressing Missing Values

- Removed `bus` types and `parts only` title statuses as they are irrelevant.
- Excluded truck models and manufacturers from the analysis.
- Assigned '8 cylinders' and 'automatic' transmission to electric cars for consistency.
- Imputed missing `manufacturer` values based on the associated `model`.
- Developed a mapping from `model` to populate missing attributes like `manufacturer`, `cylinders`, `fuel`, etc.

**Improvement in Data Completeness:**

- `year`:         0.153%
- `model`:        1.257%
- `condition`:   40.925%
- `odometer`:     0.888%
- `VIN`:         37.733%

### Handling Outliers

- Removed vehicles older than 20 years and applied the IQR method to adjust for outliers in the `price` and `odometer` data.

Before and after cleaning:
![Before Cleaning](./images/boxplot_price_year_odometer.png)
![After Cleaning](./images/boxplot_price_year_odometer_after_outliers.png)

After dropping outliers and columns we don't need, we are with zero null values.

## Data Pre-Processing

- Removed non-predictive columns such as `region`, `paint_color`, and `state`.
- Applied one-hot encoding to categorical variables like `manufacturer`, `fuel`, `transmission`, `drive`, and `type`.
- Conducted label encoding on `title_status` based on predefined values.

## Models used for prediction

1. Linear Regression
2. LinearRegression with SequentialFeatureSelector and GridSearchCV
3. LinearRegression with RobustScaler
4. LinearRegression with QuantileTransformer
5. Ridge Regression with GridSearchCV to find best alpha for ridge.
6. Ridge Regression with PolynomialFeatures of degree 2.
7. LASSO Regression with GridSearchCV to find best alpha for LASSO.
8. LASSO Regression with PolynomialFeatures oe degee 2.
9. Polynomial Regression
10. LASSO Regression with PolynomialFeatures with SequentialFeatureSelector
11. Linear Regression with polynomial features using LASSO for feature selection

