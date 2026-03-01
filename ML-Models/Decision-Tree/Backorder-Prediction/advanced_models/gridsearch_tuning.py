
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def tune_decision_tree(X_train, y_train):

    param_grid = {
        'max_depth': [4,6,8,10],
        'min_samples_split': [10,20,50],
        'min_samples_leaf': [5,10,20]
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)

    return grid.best_estimator_
