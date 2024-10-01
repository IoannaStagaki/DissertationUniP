import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_data

# Get the data after transform them
patient_data_for_diabetes = get_data()

# Separate features and target column
X_axis = patient_data_for_diabetes.drop('Outcome', axis=1)
y_axis = patient_data_for_diabetes['Outcome']

# Split the data into training and testing sets (80% training, 20% testing)
X_training, X_test, y_training, y_test = train_test_split(
    X_axis, y_axis, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_training_scaled = scaler.fit_transform(X_training)
X_test_scaler = scaler.transform(X_test)

# Define the hyperparameter grid for Logistic Regression
parameter_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs']
}

# Initialize Logistic Regression model
logistic_regression_model = LogisticRegression(random_state=42)

# Perform Grid Search Cross-Validation
grid_search_cross_validation = GridSearchCV(
    logistic_regression_model, parameter_grid, cv=5, scoring='accuracy')
grid_search_cross_validation.fit(X_training_scaled, y_training)

# Get the best model from grid search
best_model_grid_search = grid_search_cross_validation.best_estimator_

# Make predictions on the test set using the best model
y_predictions_test_set_best_model = best_model_grid_search.predict(
    X_test_scaler)

# Evaluate the model's performance
accuracy_best = accuracy_score(y_test, y_predictions_test_set_best_model)
confusion_matrix_best = confusion_matrix(
    y_test, y_predictions_test_set_best_model)
classification_report_best = classification_report(
    y_test, y_predictions_test_set_best_model)

# Display Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_best, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix With Grid Search\n')
plt.show()

# Display the best hyperparameters
print("Best Hyperparameters With Grid Search:"'\n',
      grid_search_cross_validation.best_params_)

# Display the evaluation metrics for the best model
print('\n'"Accuracy with Grid Search:{:.4f}".format(accuracy_best))
print('\nConfusion Matrix with Grid Search:''\n')
print(confusion_matrix_best)

# Display the classification report
print('\nClassification Report with Grid Search:''\n')
print(classification_report_best)
