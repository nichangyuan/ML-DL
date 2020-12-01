**Classification with XGBoost in SageMaker**

<br><br>

This is a classification project of **COVID-19** from [kaggle](https://www.kaggle.com/shirmani/characteristics-corona-patients). In the data, there are about 5% patients who die from COVID-19. The purpose of this project is to establish a machine learning model with the binary label, survival or death, and with the features like demographic factors, disease conditions, and treatment. Then deploy the model in Amazon SageMaker.

I downloaded data (version 6 - 19-5-20.csv) from kaggle.com, then worked on 
1. Pre-model Analysis: process and visualize data; get insights of features and target.
2. Problem Definition: translate the business needs into machine learning problems; design metrics to measure model performance.
3. Feature Engineering: create features which might improve model performance.
4. Modeling: Model selection, feature elimination, validation, hyperparameter tuning.

I applied 6 classification models: 1) Logistic Regression, 2) Random Forest, 3) Support Vector Machine (SVM), 4) K-Nearest Neighbors(KNN), 5) XGBoost with SMOTE/MINMAX Scaler, 6) XGBoost without SMOTE/MINMAX Scaler. I used ROC AUC as the metric to compare model performance. Below is a summary of the results.<br><br>
**ROC AUC from test data of the following models:**
1) Logistic Regression - 0.9013883266258662
2) Random Forest - 0.719504515633558
3) Support Vector Machine (SVM) - 0.8438699604696022
4) K-Nearest Neighbors(KNN) - 0.7007496041006834
5) XGBoost with SMOTE/MINMAX Scaler - 0.8660422570428405
6) XGBoost without SMOTE/MINMAX Scaler - 0.9085525414950111

    **-- For detailed analysis please refer to the notebook covid19.ipynb in current folder** 

<br><br>

From above comparison, XGBoost has the best performance. Below is the DEMO how I deployed the model using XGBoost in SageMaker. 

<br><br>

**I.** I prepared SageMaker specific data in the notebook covid19.ipynb (see it in the same folder). The generated datasets can be seen in the ***data*** subfolder, they are named *covid_train.csv and covid_test.csv*. Next, I loaded analysis data into S3.

<br><br>
**II.** Next step is the training jobs screenshot. Among them, covid-nov2020-im2 was the final trainig job set up:
<br>
<!--![](https://github.com/nichangyuan/ML-DL/blob/master/ML.jpg?raw=true)-->
<div align="center">
<img src="https://github.com/nichangyuan/ML-DL/blob/master/AWS/cov19_classification/sagemaker_snapshots/Training_jobs.PNG?raw=true" >
</div>
<br>
<br>
Below is how the hyperparameters were set up for XGBoost:<br><br>
<div align="center">
<img src="https://github.com/nichangyuan/ML-DL/blob/master/AWS/cov19_classification/sagemaker_snapshots/hyperparameters.PNG?raw=true" >
</div>
<br>
<br>

Below is the **algorithym metrics** for training and validation in XGBoost, from which it can be seen that XGBoost in SageMaker has better performance than that from previous local test in Jupyter Notebook 
**(0.96 v.s. 0.91)**
:

<br><br>
<div align="center">
<img src="https://github.com/nichangyuan/ML-DL/blob/master/AWS/cov19_classification/sagemaker_snapshots/auc.PNG?raw=true" >
</div>
<br>
<br>

**III.** Create Endpoint. <br>Below is the Endpoints:<br><br>
<div align="center">
<img src="https://github.com/nichangyuan/ML-DL/blob/master/AWS/cov19_classification/sagemaker_snapshots/Endpoints.PNG?raw=true" >
</div>
<br>
<br>

**IV.** Next, build the external Endpoint - Lambda, to communicate with internal Endpoint, also to make the gateway API for testing.

<br><br>

**V.** After these, I did testing with cURL and postman respectively, as shown below:<br><br>
<div align="center">
<img src="https://github.com/nichangyuan/ML-DL/blob/master/AWS/cov19_classification/sagemaker_snapshots/curl_test.PNG?raw=true" >
</div>
<br>
<br>
<div align="center">
<img src="https://github.com/nichangyuan/ML-DL/blob/master/AWS/cov19_classification/sagemaker_snapshots/Postman3.PNG?raw=true" >
</div>
<br>

<br>

*Below screenshot are the contents in the 3 text files which were used in cURL testing and Postman testing:*<br><br>
Each data array has 8 data points, which are in order corresponding to 'bg_disease_cnt', 'symptoms_cnt', 'days_onset', 'days_confirm', 'gender', 'age', 'smoke', 'treat'.<br><br>

- 'bg_disease_cnt' is the counts of background diseases; 
- 'symptoms_cnt' is the counts of symptoms when diagnosed as COVID-19; 
- 'days_onset' is the days from diagnosis to the analysis day (May 19, 2020);     
- 'days_confirm' is the days from the day COVID-19 confirmed to the analysis day;     
- 'gender' is 1 for female, 2 for male, 0 for missing; 
- 'age', self-explained; 
- 'smoke' is 1 for smoker, 0 for nonsmoker;   
- 'treat' is 3 if hospitalized, 2 if go to clinical, 0 if home isolation.

<br><br>
<div align="center">
<img src="https://github.com/nichangyuan/ML-DL/blob/master/AWS/cov19_classification/sagemaker_snapshots/data_test.PNG?raw=true" >
</div>
<br>
<br>

**Conclusions:**

<br>

- XGBoost model deployed in SageMaker successfully;
- XGBoost in Sagemaker has better performance comparing to the same model I established locally;
- The testing results are encouraging: curl 1 patient **survived** and result was 0.023, which is close to 0; curl 2 and curl 3 (Postman was for same data as curl 3) patients **died** and results are 0.92 and 0.84, respectively, which are close to 1. 

<br>












