from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import os
import sys
import warnings
import pandas as pd
from sklearn.metrics import classification_report
import joblib
from django.conf import settings

sys.getdefaultencoding()

flag = False  # Use a boolean flag instead of 0
query = ""
warnings.filterwarnings("ignore")


# Create your views here.
def openHome(request):
    return render(request, "cardiac.html")


def cardiacReport(request):
    return render(request, "cardiac_report.html")


def prediction(request):
    return render(request, "prediction.html")


def predict(request):
    # Provided input for the individual person's test data after 4 years
    test_data_after_4_years = {
        "Age": [int(request.POST["txtAge"])],
        "Gender_Male": [int(request.POST["selGender"])],  # Use 1 for Male, 0 for Female
        "Diabetes": [int(request.POST["selDiabetes"])],  # 0 for No, 1 for Yes
        "Hypertension": [int(request.POST["selHypertension"])],  # 0 for No, 1 for Yes
        "Kidney_Disease": [
            int(request.POST["selKidneyDisease"])
        ],  # 0 for No, 1 for Yes
        "Respiratory_Issues": [
            int(request.POST["selRespiratoryIssues"])
        ],  # 0 for No, 1 for Yes
        "Ejection_Fraction": [int(request.POST["selEjectionFraction"])],
        "Surgery_Type_CABG": [
            int(request.POST["selSurgeryTypeCABG"])
        ],  # 1 for CABG, 0 for others
        "Surgery_Type_Valve": [
            int(request.POST["selSurgeryTypeValve"])
        ],  # 1 for Valve, 0 for others
        "Surgery_Type_Congenital": [
            int(request.POST["selSurgeryTypeCongenita"])
        ],  # 1 for Congenital, 0 for others
        "Surgery_Type_Aneurysm": [
            int(request.POST["selSurgeryTypeAneurysm"])
        ],  # 1 for Aneurysm, 0 for others
        "Surgery_Type_Transplant": [
            int(request.POST["selSurgeryTypeTransplant"])
        ],  # 1 for Transplant, 0 for others
        "Complications_4_Years_Post_Surgery": [0],
    }

    # Create a DataFrame for the individual person's test data after 4 years
    individual_test_data_after_4_years = pd.DataFrame(test_data_after_4_years)

    # Load the trained model
    model = joblib.load(
        os.path.join(settings.STATICFILES_DIRS[0], "best_cardiac_surgery_model.joblib")
    )

    # Load the feature names used during training
    training_feature_names = joblib.load(
        os.path.join(settings.STATICFILES_DIRS[0], "model_feature_names.joblib")
    )

    # Select only the features present in the training data from the individual test data after 4 years
    X_test_individual_after_4_years = individual_test_data_after_4_years[
        training_feature_names
    ]

    # Make predictions on the individual person's test data after 4 years
    predictions_individual_after_4_years = model.predict(
        X_test_individual_after_4_years
    )

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

    return HttpResponse(severity_percentage_after_4_years)
