import shap
import matplotlib.pyplot as plt
import os

def run_shap_analysis(model, X_train, X_test):

    print("Running SHAP explainability...")

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("results/shap_summary.png", bbox_inches="tight")
    plt.close()

    # Feature importance plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig("results/shap_feature_importance.png", bbox_inches="tight")
    plt.close()

    print("SHAP plots saved in results/")