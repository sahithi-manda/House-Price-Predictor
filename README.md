# California House Price Prediction

## 🎯 Objective
This project is the **Final Project** simulating a real-world end-to-end Machine Learning workflow. The goal is to predict the median house value in California districts based on demographic and geographical features using supervised machine learning (regression).

## 🧠 Skills Demonstrated
- Data gathering and loading (Sklearn datasets -> CSV)
- Data cleaning and preprocessing (handling duplicates and nulls)
- Exploratory Data Analysis (EDA) using `seaborn` and `matplotlib`
- Feature scaling using `StandardScaler`
- Supervised Regression Modeling (`LinearRegression`, `RandomForestRegressor`)
- Model evaluation (RMSE, R²)
- Hyperparameter tuning (`GridSearchCV`)
- Feature Importance extraction and Business insights

## 📁 Repository Structure
- `House_Price_Prediction.ipynb`: Complete Jupyter Notebook containing the 12-step structured workflow with all code, analysis, and outputs.
- `Project_Report.md`: A detailed Markdown report summarizing the problem statement, EDA insights, model evaluation, and actionable business recommendations.
- `app.py`: Streamlit script containing an interactive web application that loads our trained Random Forest model to predict house values based on user inputs.
- `rf_model.joblib`: The trained and serialized Random Forest model used by the web app.
- `california_housing.csv`: The exported raw dataset.
- `requirements.txt`: Python package dependencies.

## 🚀 Setup and Execution
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Notebook:**
   Open `House_Price_Prediction.ipynb` in your preferred Jupyter environment (e.g., VS Code or JupyterLab) and run all cells to view the full EDA and training pipeline.
3. **Launch the Web App:**
   Run the following command to start the interactive Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
