import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, GridSearchCV, StratifiedKFold
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, make_scorer, f1_score
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel

X_data = np.load('X_data.npy', allow_pickle=True)
y_data = np.load('y_data.npy', allow_pickle=True)

print(X_data.shape)
print(y_data.shape)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42, shuffle=True)
y_test = y_test[:, 0] # won or not
y_train = y_train[:, 0] # won or not
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

unique_classes, class_counts = np.unique(y_train, return_counts=True)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict = dict(zip(np.unique(y_train), class_weights))

selector = SelectFromModel(estimator=xgb.XGBClassifier(), threshold='median')
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

print(f"Number of features selected: {X_train_selected.shape[1]}")

# XGBoost Regressor
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=weight_dict[1] / weight_dict[0]
)
print(f"Y test: {y_test[:10]}")

xgb_model.fit(X_train_selected, y_train)
y_pred = xgb_model.predict(X_test_selected)
y_pred_proba = xgb_model.predict_proba(X_test_selected)[:, 1] # probability of winning

print(f"Y pred: {y_pred[:10]}")
print(f"Y pred proba: {y_pred_proba[:10]}")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')
# print(f"Cross-validation scores: {cv_scores}")
# print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


# Create scorers
scorers = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score, average='weighted')
}

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

# Set up the grid search
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=scorers,
    refit='f1',  # You can change this to 'accuracy' if you prefer
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    verbose=2,
    n_jobs=-1
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best F1-score:", grid_search.best_score_)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred = best_model.predict(X_test)