
# **🏢 Energy Star Score Prediction**

## **📌 Introduction**
Energy Star Score Prediction is a machine learning project aimed at estimating the **Energy Star Score** of buildings based on real-world **New York City energy and water usage data**. The project helps identify the key factors influencing building energy efficiency while providing insights into greenhouse gas emissions and water consumption.

The model analyzes building attributes, energy usage, and environmental factors to **predict the Energy Star Score** using machine learning techniques such as **Gradient Boosting, Random Forest, and Linear Regression**. By leveraging **feature selection, hyperparameter tuning, and interpretability tools**, the project delivers **accurate** and **explainable** predictions.

---

## **🚀 Features**
### 🔍 **Data Processing & Feature Engineering**
- Handles **missing values, outliers, and categorical encoding**.
- **Log transformation & normalization** to enhance feature distribution.
- **Collinearity analysis** to remove redundant features.

### 📊 **Machine Learning Models**
- **Baseline Model:** Predicts using the median score.
- **Models Evaluated:**
  - **Linear Regression**
  - **Support Vector Regression (SVR)**
  - **Random Forest Regressor**
  - **Gradient Boosting Regressor (GBR)**
  - **K-Nearest Neighbors Regressor (KNN)**
- **Best Model Selected:** **Gradient Boosting Regressor (GBR)**.

### 🎯 **Performance Optimization**
- **Hyperparameter tuning** using **RandomizedSearchCV** and **GridSearchCV**.
- Final model **reduces prediction error (MAE) from 25 to ~9.1**.

### 🧐 **Model Explainability**
- **Feature Importance Analysis:** Identifies key energy factors.
- **LIME Interpretation:** Explains individual predictions.
- **Decision Tree Visualization:** Shows how the model makes decisions.

---

## **📂 Dataset**
The dataset used in this project comes from **New York City's Energy and Water Data Disclosure (2016)** and includes:
- **Building Attributes:** Property Type, Borough, Year Built, Gross Floor Area.
- **Energy Consumption:** Site EUI, Weather Normalized EUI, Electricity, Gas Usage.
- **Water Consumption:** Total Water Use, Water Intensity.
- **GHG Emissions:** Total CO₂ Emissions (Direct & Indirect).

---

## **🛠️ Installation & Setup**
### **🔹 Step 1: Clone the Repository**
```bash
git clone https://github.com/VINEEL8055/Eenrgy-Star-Score.git
cd energy-star-score-prediction
```

### **🔹 Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **🔹 Step 3: Run the Jupyter Notebook**
```bash
jupyter notebook
```
- Open `Energy_Star_Score_Prediction.py` and run the cells to train and evaluate the models.

---

## **🖥️ Usage**
1. **Load and Clean Data:** Run the notebook to clean, transform, and preprocess the dataset.
2. **Train Machine Learning Models:** Compare different models to find the best predictor.
3. **Evaluate Performance:** Measure **Mean Absolute Error (MAE)** to compare accuracy.
4. **Hyperparameter Tuning:** Optimize the best model for **higher accuracy**.
5. **Interpret Model Predictions:** Use **LIME and Feature Importance Analysis** to explain results.

---

## **📊 Results & Insights**
### ✅ **Baseline Model Performance**
- **Median Score Guess (Baseline MAE):** ~25  
- **Machine Learning Model Performance:**
  - **Gradient Boosting Regressor (Best Model) → MAE: ~9.1**
  - **Random Forest (Runner-up) → MAE: ~10.5**
  - **Other models performed worse than GBR & RF**.

### 📌 **Key Findings**
- **High Site EUI (Energy Use Intensity) → Low Energy Star Score**.
- **GHG Emissions and Energy Usage Metrics were the strongest predictors**.
- **Building type affects scores more than borough location**.

---

## **⚙️ Technologies Used**
- **Python 🐍**
- **Pandas, NumPy** (Data Processing)
- **Matplotlib, Seaborn** (Visualization)
- **Scikit-Learn** (Machine Learning)
- **LIME** (Model Explainability)

---

## **🚀 Future Improvements**
🔹 **Reduce Overfitting:**  
- Further optimize hyperparameters to **balance bias-variance trade-off**.  

🔹 **Experiment with Advanced Models:**  
- Try **XGBoost** or **Neural Networks** for better performance.  

🔹 **Deploy the Model:**  
- Convert the trained model into a **real-time API** for energy efficiency evaluation.  

🔹 **Build a Web Dashboard:**  
- Create an **interactive dashboard** for users to input building details and receive predictions.

---

## **📜 License**
This project is **open-source**.

---

## **📬 Contact & Contributions**
💡 Contributions are welcome! If you'd like to improve this project, feel free to submit a **pull request**.  

📩 For inquiries, reach out via: **vineelrayapati@gmail.com**  

---
