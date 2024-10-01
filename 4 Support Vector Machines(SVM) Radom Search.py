import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import get_data


# Get the data after transform them
patient_data_for_diabetes = get_data()

# Separate features and target column
X_axis = patient_data_for_diabetes.drop(['Outcome'], axis=1)
y_axis = patient_data_for_diabetes['Outcome']

# Split the data into training and testing sets (80% training, 20% testing)
X_trainning, X_test, y_trainning, y_test = train_test_split(
    X_axis, y_axis, test_size=0.2)

# Standardize features
scaler = StandardScaler()
X_trainning_scaler = scaler.fit_transform(X_trainning)
X_test_scaler = scaler.transform(X_test)

# Define the hyperparameter distribution for SVM
parameter_distribution_SVM = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Initialize SVM model
svm_model = SVC()

# Perform Randomized Search Cross-Validation
random_search_cross_validation = RandomizedSearchCV(
    svm_model, param_distributions=parameter_distribution_SVM, n_iter=10, cv=5, scoring='accuracy')
random_search_cross_validation.fit(X_trainning_scaler, y_trainning)

# Get the best SVM model from random search
best_svm_model_random_search = random_search_cross_validation.best_estimator_

# Make predictions on the test set using the best SVM model
y_predictions_test_set_best_svm_model = best_svm_model_random_search.predict(
    X_test_scaler)

# Evaluate the SVM model's performance
accuracy_best_svm = accuracy_score(
    y_test, y_predictions_test_set_best_svm_model)
confusion_matrix_best_svm = confusion_matrix(
    y_test, y_predictions_test_set_best_svm_model)
classification_report_best_svm = classification_report(
    y_test, y_predictions_test_set_best_svm_model)

# Display Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_best_svm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Display the best hyperparameters
print("Best Hyperparameters with Random Search:\n",
      random_search_cross_validation.best_params_)

# Display the evaluation metrics for the best SVM model
print('\n'"SVM Accuracy with Random Search:{:.4f}".format(accuracy_best_svm))
print('\nConfusion Matrix for SVM Model with Random Search:')
print(confusion_matrix_best_svm)

print('\nClassification Report for SVM Model with Random Search:''\n')
print(classification_report_best_svm)
