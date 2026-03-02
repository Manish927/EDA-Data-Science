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



def shap_waterfall_plot(model, X_test):

    print("Generating SHAP waterfall plot...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_test)

    os.makedirs("results", exist_ok=True)

    plt.figure()

    shap.plots.waterfall(shap_values[0], show=False)

    plt.savefig("results/shap_waterfall.png", bbox_inches="tight")

    plt.close()

    print("Waterfall plot saved.")


def shap_force_plot(model, X_test):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    shap.initjs()

    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_test.iloc[0]
    )

    return force_plot


def shap_decision_plot(model, X_test):

    print("Generating SHAP decision plot...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    os.makedirs("results", exist_ok=True)

    plt.figure()

    shap.decision_plot(
        explainer.expected_value,
        shap_values[:50],
        X_test.iloc[:50],
        show=False
    )

    plt.savefig("results/shap_decision_plot.png", bbox_inches="tight")

    plt.close()

    print("Decision plot saved.")