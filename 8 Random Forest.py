import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint
from utils import get_data


# Get the data after transform them
patient_data_for_diabetes = get_data()

# SSeparate features and target column
X_axis = patient_data_for_diabetes.drop(['Outcome'], axis=1)
y_axis = patient_data_for_diabetes['Outcome']

# Split the data into training and testing sets
X_trainning, X_test, y_trainning, y_test = train_test_split(
    X_axis, y_axis, test_size=0.2)

# Define parameter distributions for RandomizedSearchCV
parammeter_distribution_random_search = {
    'n_estimators': randint(5, 100),
    'max_depth': randint(1, 12)
}

# Instantiate a RandomForestClassifier
random_forest_classifier_model = RandomForestClassifier(random_state=42)

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=random_forest_classifier_model, param_distributions=parammeter_distribution_random_search, n_iter=10, cv=5, n_jobs=-1, random_state=42, verbose=0)

# Train the model with training data
random_search.fit(X_trainning, y_trainning)

# Use the best model
best_random_forest_model = random_search.best_estimator_

# Make predictions on the test set
y_predictions_test_set_best_random_forest_model = best_random_forest_model.predict(
    X_test)

# Evaluate the Random Forest model's performance
accuracy_best_random_forest = accuracy_score(
    y_test, y_predictions_test_set_best_random_forest_model)
confusion_matrix_best_svm = confusion_matrix(
    y_test, y_predictions_test_set_best_random_forest_model)


# Display Confusion Matrix
sns.heatmap(confusion_matrix_best_svm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Display the best hyperparameters
print('Best Parameters with Random Search:\n', random_search.best_params_)

# Display the evaluation metrics for the best Random Forest model
print('\n Accuracy with Random Search: {:.4f}'.format(
    accuracy_best_random_forest))
print('\nConfusion Matrix with Random Search:')
print(confusion_matrix_best_svm)

# Display the classification report
print('\nClassification Report with Random Search:\n')
print(classification_report(y_test, y_predictions_test_set_best_random_forest_model))
