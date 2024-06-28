import pandas as pd
from sklearn.metrics import classification_report
import joblib

# Provided input for the individual person's test data after 4 years
test_data_after_4_years = {
    "Age": [60],
    "Gender_Male": [0],  # Use 1 for Male, 0 for Female
    "Diabetes": [0],  # 0 for No, 1 for Yes
    "Hypertension": [1],  # 0 for No, 1 for Yes
    "Kidney_Disease": [0],  # 0 for No, 1 for Yes
    "Respiratory_Issues": [0],  # 0 for No, 1 for Yes
    "Ejection_Fraction": [55],
    "Surgery_Type_CABG": [1],  # 1 for CABG, 0 for others
    "Surgery_Type_Valve": [0],  # 1 for Valve, 0 for others
    "Surgery_Type_Congenital": [0],  # 1 for Congenital, 0 for others
    "Surgery_Type_Aneurysm": [0],  # 1 for Aneurysm, 0 for others
    "Surgery_Type_Transplant": [0],  # 1 for Transplant, 0 for others
    "Complications_4_Years_Post_Surgery": [0],
}

# Create a DataFrame for the individual person's test data after 4 years
individual_test_data_after_4_years = pd.DataFrame(test_data_after_4_years)

# Load the trained model
model = joblib.load("best_cardiac_surgery_model.joblib")

# Load the feature names used during training
training_feature_names = joblib.load("model_feature_names.joblib")

# Select only the features present in the training data from the individual test data after 4 years
X_test_individual_after_4_years = individual_test_data_after_4_years[
    training_feature_names
]

# Make predictions on the individual person's test data after 4 years
predictions_individual_after_4_years = model.predict(X_test_individual_after_4_years)

# Display classification report after 4 years
classification_result_after_4_years = classification_report(
    individual_test_data_after_4_years["Complications_4_Years_Post_Surgery"],
    predictions_individual_after_4_years,
    labels=[0, 1],
    target_names=["No Complications", "Complications"],
)
print("Classification Report for Individual Person after 4 years:")
# print(classification_result_after_4_years)

# Calculate the probability of complications after 4 years (severity)
probabilities_individual_after_4_years = model.predict_proba(
    X_test_individual_after_4_years
)[
    :, 1
]  # Probability of class 1 (Complications)

# Display severity in percentage after 4 years
severity_percentage_after_4_years = probabilities_individual_after_4_years[0] * 100
print(
    f"Severity of Complications after 4 years (in percentage): {severity_percentage_after_4_years:.2f}%"
)
