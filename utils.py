import pandas as pd
import os

# create a fuction


def get_data():
    csv = 'HealthcareDiabetes.csv'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, csv)
    data = pd.read_csv(data_path)

    # Drop specified columns
    columns_to_remove = ['Id', 'Insulin', 'SkinThickness',
                         'BloodPressure', 'DiabetesPedigreeFunction']
    data_reduced = data.drop(columns=columns_to_remove)

    # Remove rows with 0 values in all columns except 'Pregnancies' and 'Outcome'
    columns_to_validate = data_reduced.columns.difference(
        ['Pregnancies', 'Outcome'])

    patient_data_for_diabetes = data_reduced[(data_reduced[columns_to_validate] != 0)
                                             .all(axis=1)
                                             .copy()]
    return patient_data_for_diabetes
