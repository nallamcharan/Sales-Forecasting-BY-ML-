#import wanted libraries
from flask import Flask,request,render_template,redirect,url_for
import joblib
import pandas as pd 

#create a web app
app = Flask(__name__)
model = joblib.load("xgbmodel.pk1")


#run the server firtly
@app.route("/")
def index():
    return render_template('index.html')

#prediction
@app.route("/predict",methods = ["POST","GET"])

def forecast():
    product_name = request.form['pn']
    category = request.form['pc']
    price  = int(request.form['p'])
    discount  = float(request.form['d'])
    month = int(request.form['MM'])
    quantity = int(request.form['q'])
    area = request.form['City']
    new_data  = pd.DataFrame([{'Product Name':product_name,
                               'Product Category':category,
                               "Price Each":price,
                               'Quantity':quantity,
                               'Month':month,
                               'Discount':discount,
                               "City":area
                               }])
    predicted_result  = model.predict(new_data)
    result  = print(predicted_result)

    return render_template(
        'after.html',
        prediction=predicted_result,
        pn=product_name,
        pc=category,
        q=quantity,
        p=price,
        d=discount,
        city=area,
        month=month
    )
#running
if __name__ =="__main__":
    app.run(debug=True)