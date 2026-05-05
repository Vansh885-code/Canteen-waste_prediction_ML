# Canteen Food Waste Prediction 🍱

Machine Learning model to predict daily canteen food waste in kg using RandomForest Regressor. This project helps canteens reduce food wastage by up to 25% through better meal planning.

### 📊 Model Performance
- **Algorithm:** RandomForest Regressor
- **Test R2 Score:** 0.87
- **Mean Absolute Error (MAE):** 3.61 kg
- **Dataset:** 180 days synthetic data with features like weather, events, holidays

### 🔑 Key Insights from Analysis
1. **Special Events:** Exams/Fests reduce waste by ~60% due to lower student attendance
2. **Holidays:** Minimal waste generated as canteen operates with limited capacity
3. **Weather Impact:** Temperature and weather conditions have relatively low impact on waste

### 🛠️ Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib, Google Colab

### 📂 Repository Structure
1. `another_copy_of_untitled8.py` - Complete code: Data generation + EDA + Model training
2. `canteen_waste.csv` - Synthetic dataset of 180 days used for training
3. `canteen_waste_model.pkl` - Trained RandomForest model ready for deployment

### 🚀 How to Use
```python
import joblib
model = joblib.load('canteen_waste_model.pkl')
prediction = model.predict([[150, 1, 2, 0, 1, 28.5, 0]]) # Sample features
print(f"Predicted Waste: {prediction[0]:.2f} kg")
