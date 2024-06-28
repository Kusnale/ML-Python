import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to generate synthetic data for cardiac surgery risk prediction
def generate_data(num_samples):
    np.random.seed(42)

    # Demographics
    age = np.random.randint(30, 80, size=num_samples)
    gender = np.random.choice(["Male", "Female"], size=num_samples)

    # Medical history
    diabetes = np.round(np.random.uniform(70, 136, size=num_samples)).astype(int)
    hypertension = np.round(np.random.uniform(90, 180, size=num_samples)).astype(int)
    kidney_disease = np.round(np.random.uniform(10, 90, size=num_samples)).astype(int)
    respiratory_issues = np.round(np.random.uniform(1, 100, size=num_samples)).astype(
        int
    )

    # Cardiac function
    ejection_fraction = np.random.randint(30, 70, size=num_samples)

    # Type of surgery
    surgery_type = np.random.choice(
        ["CABG", "Valve", "Congenital", "Aneurysm", "Transplant"], size=num_samples
    )

    # Complications within 4 years post-surgery (1 for complications, 0 for no complications)
    complications_4_years = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])

    # Create DataFrame
    data = pd.DataFrame(
        {
            "Age": age,
            "Gender": gender,
            "Diabetes": diabetes,
            "Hypertension": hypertension,
            "Kidney_Disease": kidney_disease,
            "Respiratory_Issues": respiratory_issues,
            "Ejection_Fraction": ejection_fraction,
            "Surgery_Type": surgery_type,
            "Complications_4_Years_Post_Surgery": complications_4_years,
        }
    )

    # Convert categorical variables into numerical using one-hot encoding
    data = pd.get_dummies(data, columns=["Gender", "Surgery_Type"], drop_first=True)

    return data


# Generate a main dataset with 500 samples
main_dataset = generate_data(500)

# Group data by age and calculate the proportion of patients with complications for each age group
age_complications_proportion = main_dataset.groupby('Age')['Complications_4_Years_Post_Surgery'].mean().reset_index()

# Scale the proportion to the range of 0 to 100 percent
age_complications_proportion['Complications_4_Years_Post_Surgery'] *= 100

# Plotting age versus proportion of patients with complications
plt.figure(figsize=(10, 10))
plt.plot(age_complications_proportion['Age'], age_complications_proportion['Complications_4_Years_Post_Surgery'], marker='o')
plt.title('Age vs Proportion of Patients with Complications 4 Years Post Surgery')
plt.xlabel('Age')
plt.ylabel('Proportion of Patients with Complications (%)')
plt.grid(True)
plt.show()

