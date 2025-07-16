# 🧠 Parkinson's Disease Prediction using Machine Learning

This project demonstrates how to use machine learning techniques to predict whether a person has Parkinson’s Disease based on diagnostic biomarkers. Parkinson’s is a progressive neurological disorder characterized by tremors, stiffness, and slowed movements. While there's no definitive test for diagnosis, ML models can help make accurate predictions based on patterns in medical data.

---

## 📁 Dataset

- **Source**: Parkinson’s Disease Dataset (includes 755 features and 3 observations per patient).
- **Format**: CSV  
- **Size**: 252 aggregated patient records after processing.

---

## 🧰 Libraries Used

- `Pandas`, `NumPy`: Data manipulation  
- `Matplotlib`, `Seaborn`: Data visualization  
- `Scikit-learn`: ML algorithms and preprocessing  
- `XGBoost`: Gradient boosting classifier  
- `Imbalanced-learn`: Oversampling minority classes  
- `TQDM`: Progress bars

---

## 📊 Project Workflow

### 1. Data Import and Exploration
- Load dataset using `pandas`
- Use `df.info()`, `df.describe()`, and `df.isnull()` to explore the data
- ✅ Confirmed: No missing values

### 2. Data Wrangling
- Aggregated records per patient using `groupby('id').mean()`
- Removed `id` column
- Removed highly correlated features (`correlation > 0.7`)
- ✅ Result: 287 features retained

### 3. Feature Selection
- Normalized data with `MinMaxScaler`
- Applied Chi-Square test (`SelectKBest`) to retain top 30 features
- ✅ Result: 30 features + 1 target column (`class`)

### 4. Class Imbalance Handling
- Detected imbalance via pie chart
- Applied `RandomOverSampler` to balance classes
- ✅ Final training data: 302 samples (151 healthy / 151 unhealthy)

### 5. Model Training
Three models trained:
- Logistic Regression  
- XGBoost Classifier  
- Support Vector Machine (RBF Kernel)

Evaluated using ROC AUC Score (Training & Validation)  
✅ Best model: **Logistic Regression** (smallest gap between training and validation scores)

### 6. Model Evaluation
**Confusion Matrix for Logistic Regression**:
- TP: 35, TN: 10, FP: 4, FN: 2

**Classification Report**:
- Precision, Recall, F1-score analyzed for each class
- ⚠️ Room for improvement in unhealthy class recall

---

## 📌 Key Observations

- Logistic Regression showed the most balanced performance.
- Feature selection using Chi-Square helped reduce overfitting and dimensionality.
- Class imbalance correction was crucial for fair evaluation.

---

## 📈 Future Improvements

- Apply deep learning models or ensemble stacking.
- Tune hyperparameters for improved recall.
- Consider external datasets for generalization.

---

## ▶️ How to Run

1. **Install requirements**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run notebook or script**:
    ```bash
    jupyter notebook parkinsons_prediction.ipynb
    ```

3. **Dataset path**:
    - Place `parkinson_disease.csv` in the root directory or update path in code.

---

## 📄 License

This project is for educational and research purposes. Attribution is appreciated if reused.
