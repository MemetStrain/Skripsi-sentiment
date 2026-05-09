import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define your paths (adjust these to match your specific horizon)
config = 'configurable'
horizon = 1
model_path = f'saved_models/{config}/h{horizon}/xgboost_base/model.pkl' 
params_path = f'saved_models/{config}/h{horizon}/xgboost_base/meta.json'

# 2. Load the trained XGBoost model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 3. Load the feature names from your saved params JSON
with open(params_path, 'r') as f:
    params = json.load(f)
    feature_names = params['feature_cols']

# 4. Extract the feature importances
# For the XGBRegressor scikit-learn API, this returns the 'gain' by default
importances = model.feature_importances_

# 5. Combine into a DataFrame and sort
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display the top 10 most important features in the console
print(f"--- Top 10 Features for Horizon {horizon} ---")
print(fi_df.head(10).to_string(index=False))

# 6. Plot the feature importances
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))
sns.barplot(
    data=fi_df.head(20), # Plotting top 20 for readability
    x='Importance', 
    y='Feature', 
    palette='viridis'
)
plt.title(f'Top 20 Feature Importances (XGBoost) — Horizon {horizon}', fontsize=14, fontweight='bold')
plt.xlabel('Relative Importance (Gain)')
plt.ylabel('Features')
plt.tight_layout()
plt.show()