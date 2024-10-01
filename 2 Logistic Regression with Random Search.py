import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    X_axis, y_axis, test_size=0.2, random_state=np.random.randint(100))

# Standardize features
scaler = StandardScaler()
X_training_scaler = scaler.fit_transform(X_training)
X_test_scaler = scaler.transform(X_test)

# Define the hyperparameter distribution for Randomized Search
parameter_distribution = {
    'C': np.round(np.logspace(-4, 4, 100), 4),
    'solver': ['liblinear', 'lbfgs']
}

# Initialize Logistic Regression model
logistic_regression_model = LogisticRegression()

# Perform Randomized Search Cross-Validation
random_search_cross_validation = RandomizedSearchCV(logistic_regression_model, param_distributions=parameter_distribution,
                                                    n_iter=10, cv=5, scoring='accuracy', random_state=np.random.randint(100))
random_search_cross_validation.fit(X_training_scaler, y_training)

# Get the best model from random search
best_model_random_search = random_search_cross_validation.best_estimator_

# Make predictions on the test set using the best model
y_predictions_test_set_best_model = best_model_random_search.predict(
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
plt.title('Confusion Matrix With Random Search\n')
plt.show()

# Display the best hyperparameters
print("Best Hyperparameters With Random Search:\n",
      random_search_cross_validation.best_params_)

# Display  the evaluation metrics for the best model
print("\nAccuracy with Random Search:{:.4f}".format(accuracy_best))
print('\nConfusion Matrix With Random Search:\n')
print(confusion_matrix_best)

# Display the classification report
print('\nClassification Report With Random Search:\n')
print(classification_report_best)
