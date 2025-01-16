
import numpy as np
import pandas as pd
from flask import Flask,request,render_template


app=Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template('index.html')

data=pd.read_csv('grades_data.csv')
X=data.iloc[:,1:4].values





@app.route('/predict',methods=['GET', 'POST'])
def predict():
    
    int_features=[float(x) for x in request.form.values()]
    y=int_features.pop()
    features=[np.array(int_features)]
    print(y)
    y=int(y)
    desired=65+y-5
    if(y==4):
      desired=83
    desired=chr(desired)
    print(desired)
    Y=data.iloc[:,y].values
    print(Y)
    
    """Splitting into Training and Test set"""
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=18)

    """Feature Scaling(standardizing the data)"""

    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)

    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor(n_estimators=500,random_state=18)
    model.fit(X_train,Y_train)

    features=scaler.transform(features)
    print(features)
    prediction=model.predict(features)
    output=np.ceil(prediction)
    return render_template('index.html',prediction_text='Predicted mark for %s' %desired+' grade {}'.format(prediction))

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)