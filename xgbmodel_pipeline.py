#Importing required libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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

#features and Targets
X  = df[['Product Name','Product Category','Quantity','Price Each','Discount','City','Month']]
y  = df["Revenue"]

#define columns types for sklearn pipelines
cat = ['Product Name', 'Product Category', 'City']
num = ['Quantity', 'Price Each', 'Discount', 'Month']


#train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=10,random_state=42)

#feature scaling for  pipeline
preprocessor   = ColumnTransformer([
    ('categorical_data',OneHotEncoder(handle_unknown='ignore'),cat),
    ('numerical_data',StandardScaler(),num)
])
#model selection
xgb_model  = XGBRegressor(objective='reg:squarederror',n_estimators =200,learning_rate  = 0.05,max_depth=6,random_state = 42)

#Pipeline building
pipeline = Pipeline([
    ('processor',preprocessor),
    ('model',xgb_model)])

#training model 
pipeline.fit(X_train,y_train)

#prediction
y_pred =pipeline.predict(X_test)

#save model

#model evaluation 
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))
print("mean_squared_error:",mean_squared_error(y_test,y_pred))
print("mean_absolute_percentage_error:",mean_absolute_percentage_error(y_test,y_pred))
print("r2_score",r2_score(y_test,y_pred))



#New Data By user 
new_data = {
    "Quantity": 5,
    "Price Each": 1200,
    "Discount": 0.1,
    "Product Name": "Laptop",
    'Product Category':'Technology',
    "City": "Zavalaland",
    "Month": 6
}


new_data_df = pd.DataFrame([new_data])

#new data prediction
new_data_prediction = pipeline.predict(new_data_df)
print("Entered product feature sales:",new_data_prediction)

#save model
joblib.dump(pipeline,'xgbmodel.pk1')