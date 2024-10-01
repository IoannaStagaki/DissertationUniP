import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
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
    X_axis, y_axis, test_size=0.2)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_trainning)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using RandomizedSearchCV
parameter_distribution_KNN = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

random_search = RandomizedSearchCV(
    KNeighborsClassifier(),
    parameter_distribution_KNN,
    n_iter=50,
    cv=5,
    scoring='accuracy'
)

random_search.fit(X_train_scaled, y_trainning)

# Get the best KNN model from random search
best_model_knn_random_search = random_search.best_estimator_
best_model_knn_random_search.fit(X_train_scaled, y_trainning)

# Make predictions on the test set
y_predictions_test_set_best_knn_model = best_model_knn_random_search.predict(
    X_test_scaled)

# Evaluate the final model
accuracy_best_random = accuracy_score(
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
plt.title('Confusion Matrix with Random Search')
plt.show()


#  Display the best hyperparameters
print('Best Hyperparameters with Random Search:\n', random_search.best_params_)

# Display the evaluation metrics for the best KNN model
print('\n KNN Accuracy with Random Search:{:.4f}'.format(accuracy_best_random))
print('\nConfusion Matrix for KNN with Random Search:')
print(confusion_matrix_best)

# Display classification report
print('Classification Report for KNN with Random Search:\n')
print(classification_report_best_knn)

# Cross-validation
cross_validation_scores_random = cross_val_score(
    best_model_knn_random_search, X_train_scaled, y_trainning, cv=5)
print('Cross-Validation Scores with Random Search:')
for i, score in enumerate(cross_validation_scores_random):
    print(f'Fold {i+1}: {score:.3f}')

# Cross-Validation Scores
plt.figure(figsize=(8, 6))
barplot_random = sns.barplot(
    x=[f'Fold {i+1}' for i in range(len(cross_validation_scores_random))], y=cross_validation_scores_random)
plt.ylim(0, 1)
plt.xlabel('Cross-Validation Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores with Random Search')

# Add exact values to each bar
for i, score in enumerate(cross_validation_scores_random):
    barplot_random.text(i, score, f'{score:.3f}', ha='center', va='bottom')

plt.show()
