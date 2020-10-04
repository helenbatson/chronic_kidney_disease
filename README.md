# Chronic Kidney Disease
Classification of chronic kidney disease

Table of Contents
1. [ Introduction. ](#intro)
2. [ The Data. ](#data)
3. [ Questions. ](#questions)
4. [ Findings. ](#findings)
5. [ Conclusion. ](#conclusion)
6. [ Implementation. ](#implem)
7. [ Instructions. ](#instruct)
8. [ Conclusion. ](#conclusion)
9. [ Licensing, Authors, Acknowledgements. ](#ack)


Using a chronic kidney disease dataset from [Kaggle]https://www.kaggle.com/colearninglounge/chronic-kidney-disease, predictions have been made on patients' kidney data to determine whether there are any clear indications of chronic kidney disease.

![Credit: favpng.com](kidney_failure.png)

<a name="intro"></a>
## 1. Introduction
A few years ago during pregnancy I was placed in the high risk group due to raised blood pressure (gestational hypertension). Ten days after birth my blood pressure rose even higher and it was determined to be preeclampsia; pronounced [pree·uh·klamp·see·uh]. Left untreated preeclampsia can lead to organ failure; in particular to the liver and kidneys.

This data science project will be to investigate kidney data to predict whether a patient has chronic kidney disease. The expectation is that the results of this project could be used to benefit people who have had their blood taken in hospital; features of their blood could be used in a model to identify chronic kidney disease which may not have been previously picked up. If I knew my equivalent data I could predict my likelihood of having this disease since having preeclampsia.

A few models will be tried, starting with logistic regression since it is used to explain the relationship between one dependent binary variable (disease present or not) and one or more nominal, ordinal, interval or ratio-level independent variables. The metrics used to measure the performance of the models will be predictions on the test data, score/classification accuracy, and a confusion matrix.

<a name="data"></a>
## 2. The Data
The data has been prepared into train and test files and a set of data models have been used in the predictions.

The columns and their descriptions are as follows:

1. age - age: Age(numerical) - in years
1. bp - blood pressure: Blood Pressure(numerical) - in mm/Hg
1. sg - specific gravity Specific Gravity(nominal) - (1.005,1.010,1.015,1.020,1.025)
1. al - albumin Albumin(nominal) - (0,1,2,3,4,5)
1. su - sugar Sugar(nominal) - (0,1,2,3,4,5)
1. rbc - red blood cells Red Blood Cells(nominal) - (normal,abnormal)
1. pc - pus cell Pus Cell (nominal) - (normal,abnormal)
1. pcc - pus cell clumps Pus Cell clumps(nominal) - (present,notpresent)
1. ba - bacteria Bacteria(nominal) - (present,notpresent)
1. bgr - blood glucose random Blood Glucose Random(numerical) - in mgs/dl
1. bu - blood urea Blood Urea(numerical) - in mgs/dl
1. sc - serum creatinine Serum Creatinine(numerical) - in mgs/dl
1. sod - sodium Sodium(numerical) - in mEq/L
1. pot - potassium Potassium(numerical) - in mEq/L
1. hemo - hemoglobin Hemoglobin(numerical) - in gms
1. pcv - packed cell volume Packed Cell Volume(numerical)
1. wc - white blood cell count White Blood Cell Count(numerical) - in cells/cumm
1. rc - red blood cell count Red Blood Cell Count(numerical) - in millions/cmm
1. htn - hypertension Hypertension(nominal) - (yes,no)
1. dm - diabetes mellitus Diabetes Mellitus(nominal) - (yes,no)
1. cad - coronary artery disease Coronary Artery Disease(nominal) - (yes,no)
1. appet - appetite Appetite(nominal) - (good,poor)
1. pe - pedal edema Pedal Edema(nominal) - (yes,no)
1. ane - anemia Anemia(nominal) - (yes,no)
1. Our target: class - Class (nominal)- class - (ckd,notckd)

<a name="questions"></a>
## 3. Questions
Can we predict chronic kidney disease from the features in the dataset?
Which features are key to the predictions?
Which is the best model to use and why?

<a name="findings"></a>
## 4. Findings
### Analysis
The heatmap below shows the relationships between features and the target ('classification').
The features 'packed cell volume' and 'hemoglobin' mostly correlate with each other, and the feature 'serum creatine' most positively correlates with having chronic kidney disease.

Packed cell volume is the the volume percentage of red blood cells in blood.
Hemoglobin is a protein in your red blood cells that carries oxygen to your organs and tissues and transports carbon dioxide from your organs and tissues back to your lungs. Low hemoglobin levels means you have a low red blood cell count (anemia).

It makes sense that these show correlation as they both refer to the number of red blood cells in the blood.

![Heatmap to check feature correlation with the target](heatmap.png)

----

The frequency plot below shows that the lower the serum creatine the less likely you are to have chronic kidney disease.
This makes sense as healthy kidneys are known to filter creatinine and other waste products from our blood.

![Chronic Kidney Disease Frequency for Serum creatinine](serum_creatinine_plot.png)

----

 ![Confusion matrix](confusion_matrix.png)

### Building the models

A few models were tried to check for the best model to predict chronic kidney disease.
- Logistic Regression
- Support Vector Machines
- Linear SVC
- Decision Tree
- Random Forest
- Naive Bayes



### What happened after adding these features?
Accuracy is low after applying a random forest classifier to the data. An SGD classifier was also used, but since the train-test split produced some labels with just zeros there was no way of moving forward with this model.
Other classifers which support multi-labels will be tried in future versions of the app.

* sklearn.tree.DecisionTreeClassifier
* sklearn.tree.ExtraTreeClassifier
* sklearn.ensemble.ExtraTreesClassifier
* sklearn.neighbors.KNeighborsClassifier
* sklearn.neural_network.MLPClassifier
* sklearn.neighbors.RadiusNeighborsClassifier
* sklearn.ensemble.RandomForestClassifier
* sklearn.linear_model.RidgeClassifierCV

<a name="conclusion"></a>
## 5. Conclusion

Can we predict chronic kidney disease from the features in the dataset?
Which features are key to the predictions?
Which is the best model to use and why?


Accuracy is .... after applying a combination of parameters using a pipeline.
By adding in more features, and doing hyperparameter tuning, the accuracy ....

This may be expected due to the number of rows versus variables, though other classifiers are worth trying to increase the current accuracy.


------------------------------------------------------------------------------------------------------------------

<a name="implem"></a>
## 6. Implementation
### Technical Information

##### Pip Install
1. sklearn
1. nltk
1. plotly
1. flask
1. joblib

##### Libraries:
1. sys
1. re
1. numpy
1. pandas
1. pickle
1. sqlalchemy
1. nltk
1. sklearn
1. plotly
1. flask
1. joblib
1. json
1. sqlalchemy
1. sqlite3


### File Descriptions
1. CKD Prediction.ipynb: Jupyter notebook with a step-by-step analysis of kidney data to predict chronic kidney disease.
1. kidney_disease_train.csv: messages created by thepublic in the native language, converted into english.
1. kidney_disease_test.csv: categories into which messages fall; water, medical, etc.
1. kidney_failure.jpg: a drawing of a funtioning kidney and a failed kidney.


<a name="instruct"></a>
## 7. Instructions:

### How To Interact With The Project
Install Jupyter notebook and open the .ipynb file to run the cells within it.


<a name="conclusion"></a>
## 8. Conclusion

Chronic kidney disease can be predicted 100% from the provided dataset using the Random Forest Classifier.
The feature most correlated to this prediction is serum creatinine, which makes sense as healthy kidneys are known to filter creatinine and other waste products from our blood.

#### Reflections
The Random Forest and Decision tree classifiers performed best in the group of models used to predict chronic kidney disease from patient data.

Removing the outliers needed some reflection as a few variables were categorical and it did not make sense to remove some of those categories. Some careful consideration was required and it paid off in the performance of the logistic regression. However in the end that was not the best model as I expected. The Random Forest classifier was best due to it's ability to limit overfitting.

#### Improvements
There was not much open data found for kidney disease, but if it was found in abundance then the chosen model may change due to considerations of performance time, and more parameters would be tested to ensure the performance was optimal.


<a name="ack"></a>
## 9. Licensing, Authors, Acknowledgements

The data files were retrieved from[Kaggle]https://www.kaggle.com/colearninglounge/chronic-kidney-disease, via https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease.

Thanks to [Udacity]https://www.udacity.com/ for supporting this project.
