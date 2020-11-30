This is a classification project of **COVID-19** data from kaggle.com. In the data, there are about 5% patients who die from COVID-19. The purpose of this project is to establish a machine learning model with the binary label, survival or death, and with the features like demographic factors, disease conditions, and treatment. Then deploy the model in Amazon SageMaker.

I downloaded data (version 6 - 19-5-20.csv) from kaggle.com, then worked on 
1. Pre-model Analysis: process and visualize data; get insights of features and target.
2. Problem Definition: translate the business needs into machine learning problems; design metrics to measure model performance.
3. Feature Engineering: create features which might improve model performance.
4. Modeling: Model selection, feature elimination, validation, hyperparameter tuning.

I applied 6 classification models: 1) Logistic Regression, 2) Random Forest, 3) Support Vector Machine (SVM), 4) K-Nearest Neighbors(KNN), 5) XGBoost with SMOTE/MINMAX Scaler, 6) XGBoost without SMOTE/MINMAX Scaler. I used ROC AUC as the metric to compare model performance. Below is a summary of the results.
**ROC AUC from test data of the following models:**
1) Logistic Regression - 0.9013883266258662
2) Random Forest - 0.719504515633558
3) Support Vector Machine (SVM) - 0.8438699604696022
4) K-Nearest Neighbors(KNN) - 0.7007496041006834
5) XGBoost with SMOTE/MINMAX Scaler - 0.8660422570428405
6) XGBoost without SMOTE/MINMAX Scaler - 0.9085525414950111

** -- For detailed analysis please refer to the notebook covid19.ipynb in current folder ** 

From above comparison, XGBoost has the best performance. So I am going to load data into S3. Data were prepared in the notebook covid19.ipynb in the data folder, they are *covid_train.csv and covid_test.csv*. 










