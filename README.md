# 🔐 Problem statement :
Produc based Companies want to predict  their future  sales to increase their revenue and company growth but they can not predict future sales in human thinking way.To solve this problem we need to develop a sales forecastor. 

# 🎯Objective 
➡️ Build a sales foresting system  by using machine learing.
➡️ Deploy it by flask and create web app

# Description 
➡️ Developed  sales prediction model by machine learning "PIPE LINE"  from scratch with help of "xgboost regression model" it produced (MAPE ERROR 9% Means Model Strong) by using 30k sales records(2023-2025) but i developed this  model without pipeline,it couldn't generate good. results.then I chose ml pipeline model ..

# DEMO OF SYSTEM
![sales_forecaterwebapp](https://github.com/user-attachments/assets/5f522b35-04a2-47da-a006-f1fd86f4b57f)
![predictedsales](https://github.com/user-attachments/assets/538c2c7a-44f6-4cd6-937b-20e8544595fb)


 # ⚙️ Tech Stack

1➡️Data manipulation(python,numpy,pandas)

2➡️ Machine learning (sklearn/scikit-learn)

3➡️ visualization(matplotlib, seaborn)

4➡️ Deployment(Flask)

# Approach 
 ➡️Here are the steps i followed to develop this model...

#1️⃣ Importing required libraries 

# 2️⃣Data collection and good inspection about data

   pd.read_csv() 

   data.info(), data.head(),data.describe()

# 3️⃣Data preprocessing 

   3.1➡️Handling missing values 
    
   3.2➡️Handling duplicatedvalues..
    
   3.3➡️Outliers Handling
    
   3.4➡️ Inserting time series columns to data frame 
   

# 4️⃣ Exploratory Data Analysis 

   4.1➡️heatmap() for correlations 
     
   4.2➡️lineplot() for month vs sales relation
     
   4.3➡️boxplot() for outliers 
     
   4.4➡️barplot() for two variate  categorical visualization 


# 5️⃣ Feature Engineering 
    
   5.1➡️Feature and Target selection 
    
   5.2➡️Train_Test_split(splitting data)
    
   5.3➡️feature encoding(LabelEncoder(),OneHotEncoder()
   
   5.4 ➡️ Feature   Scaling(StandardScaler())

  
# 6️⃣ Modeling
  Model Selection (XGBRegressor)
  
  Model Training(.fit())

  Sales Prediction (.predict()

# 9️⃣ Model Evaluation (MAE,MSE,MAPE,R2score)
  Developing model is 20% but making it to generate accurate is real skill of the data scientist.

# 🔟 Deployment by flask .
  Eventually developed app by builing HTML  web pages , CSS design 
  Connected them using falsk function()
 
▶️Eventually this is model performance 

   ▶️mean_absolute_error (58.0353)
     
   ▶️mean_squared_error (4310.151)
   
   ▶️r2_score(0.991)        
   
   ▶️mean_absolute_percentage_error(0.092)


# Impact of this model on business : 

-Busienss revenue can increase 

-Inventory plannings 

-producation plannings

-mam power adjustments
