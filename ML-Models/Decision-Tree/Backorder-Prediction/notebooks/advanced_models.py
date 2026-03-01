from src.preprocessing import load_data
from src.preprocessing import preprocess_data
from src.preprocessing import split_data

from advanced_models.imbalance_handling import apply_smote
from advanced_models.gridsearch_tuning import tune_decision_tree
from advanced_models.xgboost_model import train_xgboost
from advanced_models.shap_explainability import shap_analysis
from advanced_models.shap_explainability import shap_summary_plot

from src.evaluate import evaluate_model

df = load_data("../data/backorder.csv")

df = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(df)

# Handle class imbalance
X_train_bal, y_train_bal = apply_smote(X_train, y_train)

# Train XGBoost Model
xgb_model = train_xgboost(X_train_bal, y_train_bal)

# Evaluate
evaluate_model(xgb_model, X_test, y_test)

from src.save_model import save_model
save_model(xgb_model, "../models/backorder_model.pkl")

# SHAP Explainability
shap_values, explainer = shap_analysis(
    xgb_model,
    X_train_bal,
    X_test
)

# Generate SHAP Summary Plot
shap_summary_plot(shap_values, X_test)

# This produces a graph like: Feature importance, Impact direction, Prediction contribution
# and store on results/shap_summary.png

# The SHAP will show
# Example interpretation:
# Feature	Impact
# forecast_3_month	increases backorder risk
# lead_time	longer lead time → higher risk
# inventory	low inventory → higher ris
