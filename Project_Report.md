# Final Project Report: House Price Prediction in California

## 1. Problem Statement
The objective of this project is to simulate a real-world end-to-end Machine Learning workflow to predict the median house value for California districts. Predicting house prices is a crucial task for real estate agencies, buyers, and investors to make informed data-driven decisions. This problem is treated as a Supervised Machine Learning task (Regression).

## 2. Data Summary
The dataset used is the standard **California Housing Dataset** (fetched via `sklearn.datasets`). 
- **Target Variable:** `MedHouseVal` (Median House Value in $100,000s)
- **Features:** 8 numerical features including `MedInc` (Median Income), `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, and `Longitude`.
- The dataset contains 20,640 samples with no missing values.

## 3. Exploratory Data Analysis (EDA) Highlights
- **House Value Distribution:** The target variable `MedHouseVal` is slightly right-skewed, with a cap/spike at the maximum value of 5.0 ($500,000).
- **Correlations:** `MedInc` (Median Income) showed the highest positive correlation (0.69) with House Value, indicating that districts with higher incomes tend to have more expensive houses. `Latitude` and `Longitude` showed negative correlations, representing the geographic spread of expensive coastal homes versus cheaper inland homes.

## 4. Feature Engineering
- **Scaling:** Features were scaled using `StandardScaler` to normalize distributions. This ensures distance-based metrics and gradient-based models converge efficiently and are not biased by the scale of features like `Population` vs `AveRooms`.

## 5. Modeling and Evaluation
Two primary models were trained and evaluated on an 80-20 train-test split:
1. **Linear Regression:** Baseline model yielding an R² score of 0.576 and RMSE of 0.746 (≈ $74,600).
2. **Random Forest Regressor:** A more complex ensemble method. After initial training, hyperparameter tuning via `GridSearchCV` was applied to optimize the number of trees (`n_estimators`) and tree depth (`max_depth`). The tuned Random Forest (100 trees, unlimited depth) reached R² 0.804 and RMSE 0.506 (≈ $50,600), clearly outperforming the baseline.

## 6. Business Recommendations
Based on the feature importances extracted from our final Random Forest model:
1. **Focus on Income Demographics:** `MedInc` is the single most important predictor of house value. Marketing efforts for premium listings should prioritize demographics with high median incomes.
2. **Geographical Pricing:** Location (`Latitude` and `Longitude`) strongly impacts pricing. Real estate companies should consider leveraging these geographical factors to uncover undervalued neighborhoods that might experience future growth.
3. **Age and Size are Secondary:** While `HouseAge` and `AveRooms` are relevant, they are much less impactful than Income and Location. Buyers on a budget should prioritize older homes in good locations rather than compromising on location for a newer home.

## 7. Deliverables Included
- Complete Jupyter Notebook (`House_Price_Prediction.ipynb`) mapping out the 12 steps.
- Python app script (`app.py`) for Streamlit web app deployment.
- Saved models (`rf_model.joblib`) and dataset (`california_housing.csv`).
