# Diabetes Progression Prediction using Machine Learning

This project applies multiple regression algorithms to predict **disease progression after one year** using the **Diabetes dataset from Scikit-Learn**.  
It includes a complete ML workflow — data preprocessing, visualization, feature analysis, model training, hyperparameter tuning, evaluation, and ensemble learning.

---

## Dataset  
The dataset is loaded directly from **`sklearn.datasets.load_diabetes()`** and consists of 10 normalized clinical features such as BMI, BP, six serum measurements, age, and sex.

---

## Exploratory Data Analysis (EDA)

The project performs an extensive EDA, including:

- Histograms of all input features  
- Scatter matrix visualizations (age, sex, bmi, bp)  
- Serum subset scatter plots (s1–s6)  
- Correlation matrix heatmap  
- Feature–target correlation barplot  
- Target variable distribution  

These visualizations highlight feature relationships, importance, and statistical properties crucial for model performance.

---

## Data Preprocessing

- Train–test split (`test_size = 0.2`, `random_state = 42`)  
- Standardization using **`StandardScaler`**  
- Feature scaling visualizations (post-scaling histograms)

---

## Machine Learning Models Trained

| Model                               | Description                                   |
|-------------------------------------|-----------------------------------------------|
| **Random Forest Regressor (RF)**    | Tuned via GridSearchCV with RMSE scoring      |
| **Ridge Regression(LR)**            | Alpha hyperparameter tuned via GridSearchCV   |
| **Gradient Boosting Regressor(GBR)**| Tuned across depth, learning rate, estimators |
| **Voting Regressor**                | Ensemble of RF + Ridge + GBR                  |

Each model includes:
- Cross-validation (10-fold)  
- Learning curves  
- Test-set evaluation (RMSE, MAE, R²)  

The **Voting Regressor** is used as the final serving model.

---

## Visualizations Included

- Histograms of raw and scaled features  
- Multiple scatter matrices  
- Correlation heatmaps  
- Feature correlation with target  
- Learning curves for all models  
- Actual vs Predicted scatter plot for Voting Regressor  
- Residuals vs Predicted values  
- Permutation Feature Importance plot  

These visualizations ensure the project demonstrates strong model-interpretability practices.

---
## Tech Stack

-Python
-NumPy, Pandas
-Matplotlib, Seaborn
-Scikit-Learn

---

##  Final Evaluation Metrics  

| Model                                | Cross-Val Mean Score (RMSE) ± std| Test RMSE | Test MAE | R² Score   |
|--------------------------------------|----------------------------------|-----------|----------|------------|
| **Random Forest Regressor (RF)**     | **56.97 ± 6.396**                |**52.94**  |**43.44** | **0.4709** |
| **Ridge Regression (LR)**            | **55.58 ±4.845**                 |**53.77**  | **42.81**| **0.4541** |
| **Gradient Boosting Regressor (GBR)**| **59.09 ±  6.536**               | **52.72** | **43.21**|**0.4752**  |
| **Voting Regressor**                 | **56.19 ± 5.681**                |  **52.16**| **42.61**| **0.4863** |

---

## Model Export  
The final ensemble model is saved as:
Using:
`python
joblib.dump(voting_reg, 'voting_model_diabetes_prediction.pkl')`

---
## How to Run

Clone the repository:
`git clone https://github.com/your-username/diabetes-regression-ml.git`

---
## Author  

**Lavan Kumar Konda**  
-  2nd Year Student at NIT Andhra Pradesh  
-  Passionate about Data Science, Machine Learning, and AI  
-  [LinkedIn](https://www.linkedin.com/in/lavan-kumar-konda/)
