import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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

# Standardize the features
scaler = StandardScaler()
X_training_scaled = scaler.fit_transform(X_trainning)
X_test_scaled = scaler.transform(X_test)

# Define the hyperparameter grid for KNN
parameter_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Use GridSearchCV to tune hyperparameters
grid_search_tune_hyperparameters = GridSearchCV(KNeighborsClassifier(),
                                                parameter_grid, cv=5, scoring='accuracy')
grid_search_tune_hyperparameters.fit(X_training_scaled, y_trainning)

# Get the best KNN model from grid search
best_model_knn_grid_search = grid_search_tune_hyperparameters.best_estimator_
best_model_knn_grid_search.fit(X_training_scaled, y_trainning)

# Make predictions on the test set
y_predictions_test_set_best_knn_model = best_model_knn_grid_search.predict(
    X_test_scaled)

# Evaluate the KNN model's performance
accuracy_best_knn = accuracy_score(
    y_test, y_predictions_test_set_best_knn_model)
confusion_matrix_best = confusion_matrix(
    y_test, y_predictions_test_set_best_knn_model)
classification_report_best_knn = classification_report(
    y_test, y_predictions_test_set_best_knn_model)

# Display Confusion Matrix
sns.heatmap(confusion_matrix_best, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix with Grid Search')
plt.show()

#  Display the best hyperparameters
print("Best hyperparameters with Grid Search:\n",
      grid_search_tune_hyperparameters.best_params_, '\n')

# Display the evaluation metrics for the best KNN model
print('\n'"KNN Accuracy with Grid Search:{:.4f}".format(accuracy_best_knn))
print('\nConfusion Matrix for KNN with Grid Search:')
print(confusion_matrix_best)

# Display classification report
print('\nClassification Report for KNN with Grid Search:\n')
print(classification_report_best_knn)

# Cross-validation
cross_validation_scores_grid = cross_val_score(
    best_model_knn_grid_search, X_training_scaled, y_trainning, cv=5)
print('Cross-Validation Scores with Grid Search:')
for i, score in enumerate(cross_validation_scores_grid):
    print(f'Fold {i+1}: {score:.3f}')

# Cross-Validation Scores
plt.figure(figsize=(8, 6))
barplot_grid = sns.barplot(
    x=[f'Fold {i+1}' for i in range(len(cross_validation_scores_grid))], y=cross_validation_scores_grid)
plt.ylim(0, 1)
plt.xlabel('Cross-Validation Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores with Grid Search')

# Add exact values to each bar
for i, score in enumerate(cross_validation_scores_grid):
    barplot_grid.text(i, score, f'{score:.3f}', ha='center', va='bottom')

plt.show()
