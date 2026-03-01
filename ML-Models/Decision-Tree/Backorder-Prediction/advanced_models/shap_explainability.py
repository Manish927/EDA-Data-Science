import shap
import matplotlib.pyplot as plt


def shap_analysis(model, X_train, X_test):

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)

    return shap_values, explainer


def shap_summary_plot(shap_values, X_test):

    shap.summary_plot(shap_values, X_test)

    plt.savefig("../results/shap_summary.png")

    plt.show()


def shap_force_plot(explainer, shap_values, X_test):

    shap.initjs()

    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_test.iloc[0]
    )

