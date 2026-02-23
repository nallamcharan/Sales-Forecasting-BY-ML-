#Problem statement 
Companies want to predict future sales to increase their revenue and company growth 

🎯Objective 
Building a sales forcasting system 
 
➡️ Developed sales prediction model by machine learning "PIPE LINE"  from scrath with help of "xgboost regression model" it produced (MAPE ERROR 9% Means Model Strong) by using 30k sales records(2023-2025) but i developed this  model without pipeline,it couldn't generate good. results.then I chose ml pipeline model ..

▶️Tools I used 
1➡️Data manipulation(python,numpy,pandas)

2➡️ Machine learning (sklearn/scikit-learn)

3➡️ visualization(matplotlib, seaborn)

4➡️ Deployment(Flask)

➡️Here are the steps i followed to develop this model...

1️⃣ Importing required libraries 

2️⃣Data collection and good inspection about data

3️⃣Data preprocessing 

   3.1➡️Handling missing values 
    
   3.2➡️Handling duplicatedvalues..
    
   3.3➡️Outliers Handling
    
   3.4➡️ Inserting time series columns to data frame 

4️⃣ Exploratory Data Analysis 

   ➡️heatmap() for correlations 
     
   ➡️lineplot() for month vs sales relation
     
   ➡️boxplot() for outliers 
     
   ➡️barplot() for two variate  categorical visualization 

5️⃣Feature Engineering 
    
   5.1➡️Feature and Target selection 
    
   5.2➡️Train_Test_split(splitting data)
    
   5.3➡️feature encoding(LabelEncoder(),OneHotEncoder()
   
   5.4 ➡️ Feature   Scaling(StandardScaler())

  
6️⃣Model Selection (XGBRegressor)

7️⃣Model Training(.fit())

8️⃣Sales Prediction (.predict()

9️⃣Model Evaluation (MAE,MSE,MAPE,R2score)

🔟 Deployment by flask .
 
▶️Eventually this is model performance 

   ▶️mean_absolute_error (58.0353)
     
   ▶️mean_squared_error (4310.151)
   
   ▶️r2_score(0.991)        
   
   ▶️mean_absolute_percentage_error(0.092)


Impact of this model 

-Busienss revenue can increase 

-Inventory plannings 

-producation plannings

-mam power adjustments
