import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Car Price Prediction Project\n**Goal**: Predict Selling_Price using the provided dataset (`car data.csv`)"),
    
    nbf.v4.new_markdown_cell("## Step 1: Data Loading & Cleaning"),
    nbf.v4.new_code_cell("""# ==============================
# Cell 1: Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_theme(style='darkgrid', palette='muted')
"""),
    
    nbf.v4.new_code_cell("""# ==============================
# Cell 2: Load Dataset
# ==============================
# Load dataset
try:
    df = pd.read_csv('car data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'car data.csv' not found. Please upload the dataset.")

# Display basic info
display(df.head())
print("\\nDataset Shape:", df.shape)
print("\\nDataset Info:")
display(df.info())
print("\\nDataset Describe:")
display(df.describe())
"""),

    nbf.v4.new_code_cell("""# ==============================
# Cell 3: Data Cleaning & Feature Engineering
# ==============================
# Check for missing values
print("Missing Values:\\n", df.isnull().sum())

# Check for duplicates and drop them
print("\\nDuplicates before dropping:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates after dropping:", df.duplicated().sum())

# Feature Engineering: Calculate Car Age
current_year = 2024
df['Car_Age'] = current_year - df['Year']

# Drop unnecessary columns
# 'Car_Name' is generally dropped as it creates too many categories and 'Year' is replaced by 'Car_Age'
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

display(df.head())
"""),

    nbf.v4.new_markdown_cell("## Step 2: Exploratory Data Analysis (EDA)"),
    nbf.v4.new_code_cell("""# ==============================
# Cell 4: EDA - Selling Price Distribution
# ==============================
plt.figure(figsize=(10, 5))
sns.histplot(df['Selling_Price'], kde=True, color='blue', bins=30)
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price (in Lakhs)')
plt.ylabel('Frequency')
plt.show()

# Insight: The selling price is right-skewed, meaning most cars are sold at lower prices (under 10 Lakhs).
"""),

    nbf.v4.new_code_cell("""# ==============================
# Cell 5: EDA - Car Age vs Price
# ==============================
plt.figure(figsize=(10, 5))
sns.scatterplot(x='Car_Age', y='Selling_Price', data=df, color='red', alpha=0.7)
plt.title('Car Age vs Selling Price')
plt.xlabel('Car Age (Years)')
plt.ylabel('Selling Price (in Lakhs)')
plt.show()

# Insight: As the car gets older, its selling price generally decreases.
"""),

    nbf.v4.new_code_cell("""# ==============================
# Cell 6: EDA - Categorical Variables vs Price
# ==============================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(x='Fuel_Type', y='Selling_Price', data=df, ax=axes[0])
axes[0].set_title('Fuel Type vs Selling Price')
# Insight: Diesel cars tend to have a higher median selling price than Petrol or CNG cars.

sns.boxplot(x='Seller_Type', y='Selling_Price', data=df, ax=axes[1])
axes[1].set_title('Seller Type vs Selling Price')
# Insight: Cars sold by dealers have higher prices compared to individual sellers.

sns.boxplot(x='Transmission', y='Selling_Price', data=df, ax=axes[2])
axes[2].set_title('Transmission vs Selling Price')
# Insight: Automatic transmission cars have a significantly higher selling price than manual cars.

plt.tight_layout()
plt.show()
"""),

    nbf.v4.new_code_cell("""# ==============================
# Cell 7: EDA - Correlation Heatmap
# ==============================
plt.figure(figsize=(10, 6))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Insight: Present_Price has a strong positive correlation with Selling_Price, while Car_Age has a negative correlation.
"""),

    nbf.v4.new_markdown_cell("## Step 3: Data Preprocessing"),
    nbf.v4.new_code_cell("""# ==============================
# Cell 8: Categorical Encoding (One-Hot Encoding)
# ==============================
# Convert categorical variables using One-Hot Encoding
df = pd.get_dummies(df, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

display(df.head())
"""),

    nbf.v4.new_code_cell("""# ==============================
# Cell 9: Feature Selection & Split
# ==============================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define features (X) and target (y)
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training data shape: {X_train_scaled.shape}")
print(f"Testing data shape: {X_test_scaled.shape}")
"""),

    nbf.v4.new_markdown_cell("## Step 4 & 5: Model Building & Evaluation"),
    nbf.v4.new_code_cell("""# ==============================
# Cell 10: Model Training and Evaluation
# ==============================
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, mae, rmse

# Train and collect results
results = []
for name, model in models.items():
    r2, mae, rmse = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    results.append([name, r2, mae, rmse])

# Display Results
results_df = pd.DataFrame(results, columns=['Model', 'R² Score', 'MAE', 'RMSE'])
display(results_df.sort_values(by='R² Score', ascending=False))
"""),

    nbf.v4.new_markdown_cell("## Step 6: Hyperparameter Tuning (Advanced)"),
    nbf.v4.new_code_cell("""# ==============================
# Cell 11: Hyperparameter Tuning (Random Forest)
# ==============================
from sklearn.model_selection import RandomizedSearchCV

# Parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=10, cv=5, verbose=1, random_state=42, n_jobs=-1
)

rf_random.fit(X_train_scaled, y_train)

best_rf = rf_random.best_estimator_
print("Best Random Forest Parameters:", rf_random.best_params_)

# Evaluate best model
y_pred_best = best_rf.predict(X_test_scaled)
print("\\nTuned Random Forest R² Score:", r2_score(y_test, y_pred_best))
print("Tuned Random Forest MAE:", mean_absolute_error(y_test, y_pred_best))
print("Tuned Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_best)))
"""),

    nbf.v4.new_markdown_cell("## Step 7 & 9: Final Model, Prediction Function & Bonus Plots"),
    nbf.v4.new_code_cell("""# ==============================
# Cell 12: Residuals and Feature Importance (Bonus)
# ==============================
# Residual Error Plot
plt.figure(figsize=(10, 5))
residuals = y_test - y_pred_best
sns.scatterplot(x=y_pred_best, y=residuals, color='purple', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Error Plot (Best Model)')
plt.xlabel('Predicted Selling Price')
plt.ylabel('Residuals (Actual - Predicted)')
plt.show()
# Insight: Residuals are randomly scattered around zero, indicating a good fit.

# Feature Importance Plot
plt.figure(figsize=(10, 6))
feature_importances = best_rf.feature_importances_
sns.barplot(x=feature_importances, y=X.columns, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
# Insight: Present_Price is the most dominant feature in predicting the selling price.
"""),

    nbf.v4.new_code_cell("""# ==============================
# Cell 13: Save Final Model and Create Predict Function
# ==============================
import pickle

# Save the scaler and the model
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('car_price_model.pkl', 'wb') as file:
    pickle.dump(best_rf, file)

print("Model and Scaler saved successfully as 'car_price_model.pkl' and 'scaler.pkl'")

# Define Prediction Function
def predict_car_price(Present_Price, Kms_Driven, Owner, Car_Age, Fuel_Type, Seller_Type, Transmission):
    # Map inputs to model columns
    # We need to ensure the structure matches our trained X dataframe exactly.
    input_data = {
        'Present_Price': [Present_Price],
        'Kms_Driven': [Kms_Driven],
        'Owner': [Owner],
        'Car_Age': [Car_Age],
        'Fuel_Type_Diesel': [1 if Fuel_Type == 'Diesel' else 0],
        'Fuel_Type_Petrol': [1 if Fuel_Type == 'Petrol' else 0],
        'Seller_Type_Individual': [1 if Seller_Type == 'Individual' else 0],
        'Transmission_Manual': [1 if Transmission == 'Manual' else 0]
    }
    
    # Convert to DataFrame to ensure correct column order
    input_df = pd.DataFrame(input_data, columns=X.columns)
    
    # Scale features
    with open('scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    
    input_scaled = loaded_scaler.transform(input_df)
    
    # Load model and predict
    with open('car_price_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
        
    prediction = loaded_model.predict(input_scaled)
    return round(prediction[0], 2)

# Test the prediction function
test_price = predict_car_price(Present_Price=5.59, Kms_Driven=27000, Owner=0, Car_Age=10, 
                               Fuel_Type='Petrol', Seller_Type='Dealer', Transmission='Manual')
print(f"\\nSample Prediction Output: ₹ {test_price} Lakhs")
""")
]

nb.cells.extend(cells)

with open('Car_Price_Prediction.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Jupyter Notebook 'Car_Price_Prediction.ipynb' created successfully!")
