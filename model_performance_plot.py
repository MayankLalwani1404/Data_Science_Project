import matplotlib.pyplot as plt
import numpy as np

# Model performance values (adjust as needed based on your models/results)
models = ['Mean Sales Baseline', 'Random Forest', 'XGBoost / LightGBM']
rmse = [2750, 1051, 900]  # Replace 900 with your actual boosting model RMSE
mape = [45, 57.89, 40]    # Replace 40 with actual boosting model MAPE
r2 = [0.0, 0.58, 0.65]    # Replace 0.65 with actual boosting model R²

x = np.arange(len(models))  # model locations
bar_width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - bar_width, rmse, width=bar_width, label='RMSE')
plt.bar(x, mape, width=bar_width, label='MAPE (%)')
plt.bar(x + bar_width, r2, width=bar_width, label='R² Score')
plt.xticks(x, models, rotation=25, ha='right')
plt.ylabel('Metric Value')
plt.title('Model Performance Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('model_performance_comparison.png')
plt.show()
