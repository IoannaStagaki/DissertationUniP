import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_data


# Get the data after transform them
patient_data_for_diabetes = get_data()

# Separate features and target column
X_axis = patient_data_for_diabetes.drop(['Outcome'], axis=1)
y_axis = patient_data_for_diabetes['Outcome']


# Split the data into training and testing sets (80% training, 20% testing)
X_trainning, X_test, y_trainning, y_test = train_test_split(
    X_axis, y_axis, test_size=0.2, random_state=42)

# Define the hyperparameter grid for Gradient Boosting
parameter_grid = {
    'learning_rate': [0.01, 0.03, 0.04, 0.06],
    'n_estimators': [50, 100, 150, 200]
}

# Initialize the XGBoost model
gradient_boosting_model = XGBClassifier()

# Use GridSearchCV to tune hyperparameters
grid_search_tune_hyperparameters = GridSearchCV(
    gradient_boosting_model, parameter_grid, cv=5, scoring='accuracy')
grid_search_tune_hyperparameters.fit(X_trainning, y_trainning)

# Get the best Gradient Boosting model from grid search
best_model_gradient_boosting = grid_search_tune_hyperparameters.best_estimator_

# Make predictions on the test set using the best Gradient Boosting model
y_predictions_test_set_best_gradient_boosting_model = best_model_gradient_boosting.predict(
    X_test)

# Evaluate the Gradient Boosting model's performance
accuracy_best_gradient_boosting = accuracy_score(
    y_test, y_predictions_test_set_best_gradient_boosting_model)
confusion_matrix_best = confusion_matrix(
    y_test, y_predictions_test_set_best_gradient_boosting_model)
classification_report_best_svm = classification_report(
    y_test, y_predictions_test_set_best_gradient_boosting_model)

# Display Confusion Matrix
sns.heatmap(confusion_matrix_best, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Display the best hyperparameters
print("Best Hyperparameters with Grid Search:"'\n',
      grid_search_tune_hyperparameters.best_params_)

# Display the evaluation metrics for the best Gradient Boosting model
print("\nAccuracywith Grid Search:{:.4f}".format(
    accuracy_best_gradient_boosting))
print('\nConfusion Matrix with Grid Search:')
print('\n', confusion_matrix_best)

# Display classification report
print("\nClassification Report with Grid Search:\n")
print(classification_report_best_svm)
