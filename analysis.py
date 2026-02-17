#Importing required libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from xgboost import XGBClassifier

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
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)

##Exploratory Data Analysis
#Outliers 
from scipy.stats import zscore
outliers = df.select_dtypes(include='number').apply(zscore)
outliers1 = abs(outliers)>3
df.select_dtypes(include='number').boxplot()
plt.title("outliers")
plt.xticks(rotation=20,ha='right')
plt.savefig('outliers.jpg')
plt.show()

#which city has high sales
highsales = df.groupby("City")['Revenue'].mean().sort_values(ascending = False).head(10)
sns.barplot(x=highsales.index,y=highsales.values,palette='muted')
plt.title("Top sales based on city")
plt.xticks(rotation = 20,ha='right')
plt.savefig('cityvsrevenue.jpg')

plt.show()

#which product has hig sales 
productsales = df.groupby("Product Name")['Revenue'].mean().sort_values(ascending = False).head(10)

sns.barplot(x=productsales.index,y=productsales.values,palette='deep')
plt.title("Product vs Revenue")
plt.xticks(rotation = 20,ha='right')
plt.savefig('provsrevenue.jpg')

plt.show()

#Time series analysis
df['Order_data'] = pd.to_datetime(df['Order Date'],format='mixed',dayfirst=True)
df['Month'] = df['Order_data'].dt.month
sns.lineplot(x='Month',y='Revenue',data = df,color='gray')
plt.title("Month vs Sales")
plt.xticks(rotation=45,ha='left')
plt.savefig('Trend.jpg')

plt.show()

#correlations 
corr = df.corr(numeric_only=True)
sns.heatmap(corr,cmap="Spectral")
plt.title("Correlations")
plt.savefig('correlations.jpg')

plt.show()