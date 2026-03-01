from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model
