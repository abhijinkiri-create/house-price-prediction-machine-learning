# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts house prices using Machine Learning based on various property features such as location, overall quality, living area, garage capacity, basement area, and year built.

The project follows a complete Machine Learning workflow including:

- Data Exploration (EDA)
- Data Preprocessing
- Feature Engineering
- Model Training
- Hyperparameter Tuning
- Feature Selection
- Model Evaluation
- Prediction
- Machine Learning Pipeline

---

# 📂 Project Structure

```
House_Price_Prediction/

│
├── data/
│     ├── train.csv
│     ├── test.csv
│     └── sample_submission.csv
│
├── notebook/
│     └── EDA.ipynb
│
├── src/
│     ├── data_preprocessing.py
│     ├── model.py
│     ├── hyperparameter_tuning.py
│     ├── feature_importance.py
│     ├── feature_selection.py
│     ├── model_comparison.py
│     ├── ml_pipeline.py
│     └── predict.py
│
├── models/
│     └── house_price_pipeline.pkl
│
├── outputs/
│     ├── submission.csv
│     ├── feature_importance.csv
│     └── model_comparison.csv
│
├── requirements.txt
│
└── README.md
```

---

# 📊 Dataset

The project uses the House Prices dataset.

Target Variable:

```
SalePrice
```

Total Training Samples

```
1460
```

Testing Samples

```
1459
```

Features

```
80
```

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib

---

# 📈 Machine Learning Workflow

```
Load Dataset
      ↓
Exploratory Data Analysis
      ↓
Data Cleaning
      ↓
Missing Value Handling
      ↓
Categorical Encoding
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
Model Training
      ↓
Hyperparameter Tuning
      ↓
Feature Selection
      ↓
Model Evaluation
      ↓
Prediction
      ↓
Submission File
```

---

# 🤖 Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

# 📏 Evaluation Metrics

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

# 🚀 How to Run

Clone the repository

```bash
git clone <repository-url>
```

Open the project

```bash
cd House_Price_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python src/ml_pipeline.py
```

Generate predictions

```bash
python src/predict.py
```

---

# 📁 Output

The project generates

- submission.csv
- feature_importance.csv
- model_comparison.csv

---

# 📚 Key Skills Demonstrated

- Data Analysis
- Data Visualization
- Data Cleaning
- Feature Engineering
- Machine Learning
- Model Evaluation
- Hyperparameter Tuning
- Feature Selection
- Pipeline Development
- Regression Analysis

---

# 🔮 Future Improvements

- XGBoost
- LightGBM
- CatBoost
- SHAP Explainability
- Cross Validation Optimization
- Model Deployment using FastAPI
- Docker
- MLflow

---

# 👨‍💻 Author

**JINKIRI ABHINAY**

B.Tech CSE (AI & ML)

Machine Learning Enthusiast
