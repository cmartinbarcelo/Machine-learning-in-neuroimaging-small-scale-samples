# Machine-learning-in-neuroimaging-small-scale-samples
Code to:
- Perform Exploratory Data Analysis.
- Apply several classification models (XGBoost, kNN, SVM and RF) and study metrics variability across folds of stratified 5-fold cross validation
- Generate synthetic data with TVAE and compute performance curves according to sample size

The code assumes the name of variables, that are the output of freesurfer, but it can be changed. Also, it assumes that database is inside working directory in a folder named "BBDD", named after BBDD_{project_name}.xlsx, i.e BBDD_AD.xlsx 
