import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_data

# 1 Create percentage a pie chart for diabet diagnoses


def createPercentagePieChartForDiabetDiagnosis(outcome_counts, outcome_percentages):

    plt.figure(figsize=(10, 6))
    labelsPieChart = [f'{label} ({percentage:.1f}%)' for label, percentage in zip(
        ['Non-Diabetic', 'Diabetic'], outcome_percentages)]

    plt.pie(outcome_counts, labels=labelsPieChart, autopct='%1.1f%%',
            startangle=140, colors=['skyblue', 'grey'])
    plt.title('Count of people with and without diabetes\n')
    plt.axis('equal')
    plt.legend(title='Outcome', loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.show()

# 2 Creat a total count plot for diabet diagnoses


def createCountPlotForDiabeteDiagnose():

    plt.figure(figsize=(16, 8))
    axisCountPlot = sns.countplot(
        x='Outcome', data=patient_data_for_diabetes, palette="Set2")
    plt.title('How Many People Have Diabetes and How Many Do Not\n')

    for itemsCountPlot in axisCountPlot.patches:
        height = itemsCountPlot.get_height()
        axisCountPlot.text(itemsCountPlot.get_x() + itemsCountPlot.get_width() / 2., height + 0.05,
                           f'{height:.0f}', ha="center", va="bottom")
    plt.xlabel('Diabetes Diagnosis')
    plt.ylabel('Count of People\n')
    plt.xticks(ticks=[0, 1.0], labels=['Non-Diabetic', 'Diabetic'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 3 Creat a box plot for age distribution by diabetes diagnosis


def createBoxPlotAgeForDiabetDiagnosis(DiabetData):

    plt.figure(figsize=(10, 6))
    plt.title('Age Distribution by Diabetes Diagnosis\n')
    sns.boxplot(x='Outcome', y='Age', data=DiabetData, palette="Set2")
    plt.xlabel('Diabetes Diagnosis')
    plt.ylabel('Age of People\n')
    plt.xticks(ticks=[0, 1.0], labels=['Non-Diabetic', 'Diabetic'])
    plt.yticks(np.arange(DiabetData['Age'].min(),
               DiabetData['Age'].max() + 5, 5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 4 Creat a count plot of Age Group by Diabetes Diagnosis


def createCountPlotAgeGroupByDiabetDiagnosis(DiabetData):

    age_bins = [20, 29, 39, 49, 59, 69, 79, 89]
    age_group = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
    DiabetData['AgeGroup'] = pd.cut(
        DiabetData['Age'], bins=age_bins, labels=age_group, right=False)

    plt.figure(figsize=(18, 6))
    axisCountPlot = sns.countplot(x='AgeGroup', hue='Outcome',
                                  data=DiabetData, palette="Set2")
    plt.title('Distribution of AgeGroup by Diabetes Diagnosis')
    plt.xlabel('Number of AgeGroup')
    plt.ylabel('Count of People')
    plt.legend(title='Outcome', loc='upper right',
               labels=['Non-Diabetic', 'Diabetic'], bbox_to_anchor=(1.13, 1))
    for itemsCountPlot in axisCountPlot.patches:
        height = itemsCountPlot.get_height()
        axisCountPlot.annotate(f'{height:.0f}', xy=(itemsCountPlot.get_x() + itemsCountPlot.get_width() / 2, height),
                               xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')
    plt.ylim(0, 1100)
    plt.yticks(np.arange(0, 1101, 50))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# 5 Create a Scatter Plot between Age and Glucose Levels Based on Diabetes Diagnosis
def createScatterPlotAgeGlugoseByDiabeteDiagnosis(DiabetData):

    plt.figure(figsize=(15, 6))
    plt.title('Glucose Levels by Age and Diabetes Diagnosis')
    sns.scatterplot(x="Age", y="Glucose", hue="Outcome",
                    data=DiabetData, palette="Set2")
    plt.legend(title='Outcome', labels=[
        'Diabetic', 'Non-Diabetic'], loc='upper right', bbox_to_anchor=(1.13, 1))
    plt.yticks(np.arange(DiabetData['Glucose'].min(),
               DiabetData['Glucose'].max() + 20, 20))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 6 Create a Box Plot visualizing based on the weight and age of the people


def createBoxPlotAgeBMICategory(DiabetData):

    plt.figure(figsize=(15, 8))
    plt.title('Age by BMI Category')
    sns.boxplot(x='BMI_Full', y='Age', hue='BMI_Full',
                data=DiabetData, palette="Set2")
    plt.legend(title='BMI Category', loc='upper right',
               bbox_to_anchor=(1.13, 1))
    plt.xticks(ticks=[-0.3, 0.9, 2.1, 3.3], labels=['Obese',
                                                    'Overweight', 'Healthy', 'Underweight'])
    plt.yticks(np.arange(DiabetData['Age'].min(),
               DiabetData['Age'].max() + 2, 2))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# 7 Create a Count Plot with count of People in Each BMI Category

def createCountPlotCountPeopleBMICategory(DiabetData):

    plt.figure(figsize=(10, 6))

    axisCountPlot = sns.countplot(
        x='BMI_Full', data=DiabetData, palette="Set2")

    for itemsCountPlot in axisCountPlot.patches:
        height = itemsCountPlot.get_height()
        axisCountPlot.text(itemsCountPlot.get_x() + itemsCountPlot.get_width() / 2., height + 0.05,
                           f'{height:.0f}', ha="center", va="bottom")

    plt.title('Count of People in Each BMI Category')
    plt.xlabel('\nBMI Category')
    plt.ylabel('Count of People\n')
    plt.yticks(
        np.arange(0, DiabetData['BMI_Full'].value_counts().max() + 90, 90))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 8 Create a pie chart with BMI Categories


def createPieChartBMICategories(DiabetData):

    bmi_counts = DiabetData['BMI_Full'].value_counts().sort_index()
    bmi_percentages = (bmi_counts / bmi_counts.sum()) * 100
    labelsPieChart = [f'{label} ({percentage:.1f}%)' for label,
                      percentage in zip(bmi_counts.index, bmi_percentages)]

    plt.figure(figsize=(10, 6))
    plt.pie(bmi_counts, labels=labelsPieChart, autopct='%1.1f%%', startangle=140,
            colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    plt.title('Distribution of BMI Categories\n')
    plt.legend(title='BMI_Full', loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.axis('equal')
    plt.show()

# 9 Create a Count Plot for BMI Categories by Diabetes Diagnosy


def createCountPlotBMIDCategoriesiabetesDiagnosis(DiabetData):

    plt.figure(figsize=(15, 6))

    axisCountPlot = sns.countplot(
        x='BMI_Full', hue='Outcome', data=DiabetData, palette="Set2")

    plt.title('Distribution of BMI Categories by Diabetes Diagnosis\n')
    plt.xlabel('BMI Category')
    plt.ylabel('Count of People')
    plt.legend(title='Diabetes Diagnosis', loc='upper right',
               labels=['Non-Diabetic', 'Diabetic'])

    for itemsCountPlot in axisCountPlot.patches:
        height = itemsCountPlot .get_height()
        axisCountPlot.annotate(f'{height:.0f}', xy=(itemsCountPlot .get_x() + itemsCountPlot .get_width() / 2, height),
                               xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')
    plt.legend(title='Outcome', loc='upper right',
               labels=['Non-Diabetic', 'Diabetic'], bbox_to_anchor=(1.13, 1))
    plt.yticks(
        np.arange(0, DiabetData['BMI_Full'].value_counts().max() + 30, 30))
    plt.ylim(0, 980)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# 10 Create a Βox Plot for Glucose Levels by BMI Category

def createBoxPlotGlucoseBMICategories(DiabetData):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='BMI_Full', y='Glucose', data=DiabetData, palette="Set2")
    plt.title('Glucose Levels by BMI Category\n')
    plt.yticks(np.arange(DiabetData['Glucose'].min(),
               DiabetData['Glucose'].max() + 5, 5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 11 Create a Scatter plot for Glucose Levels by BMI Categories


def createScatterPlotGlucoseBMICategories(DiabetData):

    plt.figure(figsize=(15, 8))
    sns.scatterplot(x='BMI', y='Glucose',
                    hue='BMI_Full', data=DiabetData, palette="Set2")
    plt.title('Scatter Plot of Glucose Levels by BMI Categories')
    plt.legend(title='BMI Categories', loc='upper right',
               bbox_to_anchor=(1.13, 1))
    plt.yticks(np.arange(DiabetData['Glucose'].min(),
               DiabetData['Glucose'].max() + 20, 20))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 12 Create a Boxplot for Pregnancies by Diabetes Diagnosis


def createBoxPlotPregnaciesDiabetesDiagnosis(DiabetData):
    plt.figure(figsize=(10, 6))
    plt.title('Pregnancies by Diabetes Diagnosis\n')
    sns.boxplot(x='Outcome', y='Pregnancies', data=DiabetData, palette="Set2")
    plt.xticks(ticks=[0, 1.0], labels=['Non-Diabetic', 'Diabetic'])
    plt.yticks(np.arange(DiabetData['Pregnancies'].min(
    ), DiabetData['Pregnancies'].max() + 1, 1))
    plt.ylabel('Pregnancies\n')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# 13 Create a Count plot distribution of Pregnancies by Diabetes Diagnosis

def createCountPlotPregnanciesDiabetesDiagnosis(DiabetData):

    plt.figure(figsize=(15, 8))
    axisCountPlot = sns.countplot(x='Pregnancies', hue='Outcome',
                                  data=DiabetData, palette="Set2")
    plt.title('Distribution of Pregnancies by Diabetes Diagnosis\n')
    plt.xlabel('Pregnancies')
    plt.ylabel('Count of People')
    plt.legend(title='Diabetes Diagnosis', loc='upper right',
               labels=['Non-Diabetic', 'Diabetic'])
    for itemsCountPlot in axisCountPlot.patches:
        height = itemsCountPlot.get_height()
        axisCountPlot.annotate(f'{height:.0f}', xy=(itemsCountPlot.get_x() + itemsCountPlot.get_width() / 2, height),
                               xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')
    plt.legend(title='Outcome', labels=[
        'Non-Diabetic', 'Diabetic'], loc='upper right', bbox_to_anchor=(1.13, 1))
    plt.yticks(
        np.arange(0, DiabetData['Pregnancies'].value_counts().max() + 10, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 380)
    plt.show()

# 14 Create a Box Plot for Glucose Levels by Diabetes Diagnosis


def createBoxPlotGlucoseDiabetesDiagnosis(DiabetData):

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Outcome', y='Glucose', data=DiabetData, palette="Set2")
    plt.title('Glucose Levels by Diabetes Diagnosis\n')
    plt.xlabel('Diabetes Diagnosis')
    plt.ylabel('Glucose Levels (mg/dL)')
    plt.xticks(ticks=[0, 1], labels=['Non-Diabetic', 'Diabetic'])
    plt.yticks(np.arange(DiabetData['Glucose'].min(),
               DiabetData['Glucose'].max() + 5, 5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 220)
    plt.show()


# 15 Create a Violin Plot for Glucose Levels by Diabetes Diagnosis

def creareViolinPlotGlucoseDiabetesDiagnosis(DiabetData):
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Outcome', y='Glucose', data=DiabetData, palette="Set2")
    plt.title('Glucose Levels by Diabetes Diagnosis')
    plt.xlabel('Outcome')
    plt.ylabel('Glucose Levels (mg/dL)')
    plt.xticks(ticks=[0, 1], labels=['Non-Diabetic', 'Diabetic'])
    plt.yticks(np.arange(DiabetData['Glucose'].min(),
               DiabetData['Glucose'].max() + 4, 4))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":

    # Get the data after transform them
    patient_data_for_diabetes = get_data()

    # Calculate the counts and percentages of each Outcome
    outcome_counts = patient_data_for_diabetes['Outcome'].value_counts(
    ).sort_index()
    outcome_percentages = (outcome_counts / outcome_counts.sum()) * 100
    outcome_data = pd.DataFrame(
        {'Outcome': outcome_counts.index,
         'Count of People': outcome_counts.values,
         'Percentage': outcome_percentages})

    # Separation by weight
    patient_data_for_diabetes['BMI_Full'] = None

    for index, row in patient_data_for_diabetes.iterrows():
        bmi = row['BMI']
        if bmi > 0 and bmi < 18.5:
            patient_data_for_diabetes.at[index, 'BMI_Full'] = 'Underweight'
        elif bmi >= 18.5 and bmi <= 24.9:
            patient_data_for_diabetes.at[index, 'BMI_Full'] = 'Healthy'
        elif bmi > 24.9 and bmi < 29.9:
            patient_data_for_diabetes.at[index, 'BMI_Full'] = 'Overweight'
        else:
            patient_data_for_diabetes.at[index, 'BMI_Full'] = 'Obese'

    # 1
    createPercentagePieChartForDiabetDiagnosis(
        outcome_counts, outcome_percentages)
    # 2
    createCountPlotForDiabeteDiagnose()
    # 3
    createBoxPlotAgeForDiabetDiagnosis(patient_data_for_diabetes)
    # 4
    createCountPlotAgeGroupByDiabetDiagnosis(patient_data_for_diabetes)
    # 5
    createScatterPlotAgeGlugoseByDiabeteDiagnosis(patient_data_for_diabetes)
    # 6
    createBoxPlotAgeBMICategory(patient_data_for_diabetes)
    # 7
    createCountPlotCountPeopleBMICategory(patient_data_for_diabetes)
    # 8
    createPieChartBMICategories(patient_data_for_diabetes)
    # 9
    createCountPlotBMIDCategoriesiabetesDiagnosis(patient_data_for_diabetes)
    # 10
    createBoxPlotGlucoseBMICategories(patient_data_for_diabetes)
    # 11
    createScatterPlotGlucoseBMICategories(patient_data_for_diabetes)
    # 12
    createBoxPlotPregnaciesDiabetesDiagnosis(patient_data_for_diabetes)
    # 13
    createCountPlotPregnanciesDiabetesDiagnosis(patient_data_for_diabetes)
    # 14
    createBoxPlotGlucoseDiabetesDiagnosis(patient_data_for_diabetes)
    # 15
    creareViolinPlotGlucoseDiabetesDiagnosis(patient_data_for_diabetes)

pass
