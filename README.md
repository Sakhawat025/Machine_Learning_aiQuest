# Regression Analysis in Machine Learning
Regression analysis is one of the foundational techniques in **machine learning** that deals with predicting continuous numerical values. It is a supervised learning method where the goal is to model the relationship between a **dependent variable (target)** and one or more **independent variables (features)**. Regression analysis can help us understand the relationship between different variables and make predictions about unknown values based on this relationship.

In simple terms, regression helps answer questions like:
- How much will the price of a house increase if the number of bedrooms increases?
- What will the stock price of a company be tomorrow based on historical data?
- How much energy will be consumed based on weather conditions, household size, and other factors?


## Why is Regression Important in Machine Learning?

1. **Predicting Continuous Values**: Unlike classification problems, which deal with predicting categorical values (e.g., spam or not spam), regression focuses on predicting **continuous values**. This is particularly useful in fields like economics, finance, healthcare, and many others.
   
2. **Understanding Relationships**: Regression helps us understand the **relationship** between dependent and independent variables. For example, in predicting house prices, regression can help us quantify how much factors like the size of the house, number of rooms, and location affect the price.

3. **Model Evaluation**: Regression techniques also provide us with ways to evaluate how well our models are performing, using metrics like **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R Squared (R²)**, which tell us how well the predicted values match the actual values.


## Key Types of Regression in Machine Learning

### 1. **[Linear Regression](./2.Linear%20Regression/README.md)**
Linear Regression is used to predict a dependent variable using a linear relationship with one or more independent variables.

- **Simple Linear Regression**: Involves a single independent variable.
- **Multiple Linear Regression**: Uses multiple independent variables to predict the dependent variable.

### 2. **[Polynomial Regression](./polynomial-regression/README.md)**
Polynomial Regression extends Linear Regression by adding polynomial terms to the regression equation. It is used to model nonlinear relationships between the target and features.

### 3. **[Ridge & Lasso Regression](./ridge-lasso-regression/README.md)**
Both **Ridge** and **Lasso Regression** are regularization techniques that add a penalty term to the regression model in order to prevent overfitting and help in selecting important features.
- **Ridge Regression**: Adds an L2 penalty term (squared values of coefficients).
- **Lasso Regression**: Adds an L1 penalty term (absolute values of coefficients), which can drive some coefficients to zero, effectively selecting the most important features.

### 4. **[Logistic Regression](./logistic-regression/README.md)**
**Logistic Regression** is used for classification tasks, where the dependent variable is categorical. It can be used for binary or multi-class classification by predicting the probability of an event occurring.


## Loss and Cost Functions
Loss and cost functions are used to evaluate the performance of a regression model. Common functions include:

- **Mean Absolute Error (MAE)**: Measures the average of the absolute errors between the predicted and actual values.
- **Mean Squared Error (MSE)**: Measures the average of the squared differences between the predicted and actual values.
- **Root Mean Squared Error (RMSE)**: The square root of the MSE, providing a more interpretable value in terms of the target variable's units.


## Common Use Cases of Regression Analysis

- **Predicting House Prices**: Using features like square footage, number of bedrooms, location, etc., to predict the price of a house.
- **Stock Price Prediction**: Using historical stock data to forecast future stock prices.
- **Sales Forecasting**: Retailers use regression to predict future sales based on past sales data and external factors like advertising spend.
- **Healthcare**: Predicting patient outcomes or recovery time based on factors like age, health history, and type of treatment.
- **Energy Consumption**: Estimating the energy usage of households based on weather data, household size, and appliance usage.


## Key Evaluation Metrics for Regression Models

1. **Mean Absolute Error (MAE)**: Measures the average of the absolute errors between the predicted and actual values. It’s a simple way to understand the model's prediction accuracy.
   
2. **Mean Squared Error (MSE)**: Measures the average of the squared differences between predicted and actual values. It penalizes larger errors more than MAE.

3. **Root Mean Squared Error (RMSE)**: The square root of MSE, which brings the error metric back to the original scale of the target variable, making it more interpretable.

4. **R-squared (R²)**: A statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher R² value indicates a better fit of the model to the data.

