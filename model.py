#Importing required libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
import joblib
#Data Understanding And Collection
df  = pd.read_csv('synthetic_sales_data.csv')
print(df.info())
print(df.describe())
print(df.head())

##Data preprocessing
#Handling numerical missing values
df.fillna({'Total Revenue':df['Total Revenue'].min(),'Profit':df['Profit'].mean(),
           'Profit Margin':df['Profit Margin'].mean(),
           'Revenue':df['Revenue'].mean(),
           'Price Each':df['Price Each'].median(),
           'Discount':df['Discount'].median(),'Quantity':df['Quantity'].median()},inplace=True)
#Handling text missing values 
df['City'].fillna(df['City'].mode()[0],inplace = True)
df['Customer Name'].fillna(df['Customer Name'].mode()[0],inplace = True)
df['State'].fillna(df['State'].mode()[0],inplace = True)
df['Product Category'].fillna(df['Product Category'].mode()[0],inplace = True)
df['Area'].fillna(df['Area'].mode()[0],inplace = True)
df['Sub-Category'].fillna(df['Sub-Category'].mode()[0],inplace = True)
df['Product Name'].fillna(df['Product Name'].mode()[0],inplace = True)
print(df.isnull().sum())
#Handling duplicated values
df.drop_duplicates(inplace=True)
#adding year,month columns to data frame
df['Order_date'] = pd.to_datetime(df['Order Date'],format='mixed',dayfirst=True)
df['Month'] = df['Order_date'].dt.month
df['Year'] = df['Order_date'].dt.year

##modeling
#features and Targets
X  = df[['Product Name','Product Category','Quantity','Price Each','Discount','City','Month']]
y  = df["Revenue"]

#Label Encoding and One Hot Enconding
X_encoded = pd.get_dummies(X)

#train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_encoded,y,test_size=10,random_state=42)
#feature scaling
scaler  = StandardScaler()
X_train_scaled  =scaler.fit_transform(X_train)
X_test_scaled  =scaler.fit_transform(X_test)

#model selection
xgb_model  = XGBRegressor(objective='reg:squarederror',n_estimators =200,learning_rate  = 0.05,max_depth=6,random_state = 42)

#training model 
xgb_model.fit(X_train_scaled,y_train)

#prediction
y_pred = xgb_model.predict(X_test_scaled)

#model evaluation 
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))
print("mean_squared_error:",mean_squared_error(y_test,y_pred))
print("mean_absolute_percentage_error:",mean_absolute_percentage_error(y_test,y_pred))
print("r2_score",r2_score(y_test,y_pred))

#New Data By user 
input_row = {
    # numeric features
    "Quantity": 10,
    "Price Each": 1200,
    "Discount": 0.10,

    # product name (ONE HOT)
    "Product Name_Copy Paper Pack": 0,
    "Product Name_Dining Table": 0,
    "Product Name_Laptop": 0,
    "Product Name_Office Chair": 0,
    "Product Name_Ring Binder": 0,
    "Product Name_Smartphone": 1,
    "Product Name_Storage Box": 0,

    # city (ONE HOT)
    "City_Zavalabury": 0,
    "City_Zavalafort": 0,
    "City_Zavalaland": 1,
    "City_Zavalaview": 0,
    "City_Zhangport": 0,
    "City_Zimmermanburgh": 0,
    "City_Zimmermanbury": 0,
    "City_Zimmermanmouth": 0,
    "City_Zunigahaven": 0,

    'Month':3
}

input_data = input("enter dictionary:")
new_data = pd.DataFrame([input_row])
new_data = new_data.reindex(columns = X_train.columns)
new_data_scaled  = scaler.fit_transform(new_data)

#new data prediction
new_data_prediction = xgb_model.predict(new_data_scaled)
print("Entered product feature sales:",new_data_prediction)
