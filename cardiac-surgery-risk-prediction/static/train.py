import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the datasets
main_dataset = pd.read_csv("main_cardiac_surgery_data.csv")
dataset_scenario_2 = pd.read_csv("scenario_2_cardiac_surgery_data.csv")
dataset_scenario_3 = pd.read_csv("scenario_3_cardiac_surgery_data.csv")

# Combine the datasets
combined_dataset = pd.concat(
    [main_dataset, dataset_scenario_2, dataset_scenario_3], ignore_index=True
)

# Separate features (X) and target variable (y)
X = combined_dataset.drop("Complications_4_Years_Post_Surgery", axis=1)
y = combined_dataset["Complications_4_Years_Post_Surgery"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define hyperparameter grid for GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Perform grid search using GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Save the best model to a file
joblib.dump(best_model, "best_cardiac_surgery_model.joblib")

# Save the feature names to a separate file
joblib.dump(X.columns.tolist(), "model_feature_names.joblib")

# Make predictions on the test set using the best model
predictions = best_model.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of the Best Model: {accuracy:.2f}")

# Display classification report for the best model
classification_result = classification_report(y_test, predictions)
# print("Classification Report for the Best Model:")
# print(classification_result)
