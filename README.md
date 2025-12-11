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

##  Final Evaluation Metrics  

| Model             | RMSE | MAE | R² Score |
|-------------------|------|------|-----------|
| Random Forest     |  |  |  |
| Ridge Regression  |  |  |  |
| Gradient Boosting |  |  |  |
| Voting Regressor  |  |  |  |

---

## Model Export  
The final ensemble model is saved as:

