import shap
import matplotlib.pyplot as plt


def shap_analysis(model, X_train, X_test):

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)

    return shap_values, explainer
