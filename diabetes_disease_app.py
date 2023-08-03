import numpy as np
import pickle
from flask import Flask, request, render_template

# Load ML model
#modelgb = pickle.load(open('modelsv.pkl', 'rb'))
#modelrf = pickle.load(open('modelrf.pkl', 'rb'))
#modelknn = pickle.load(open('modelknn.pkl', 'rb'))
#modelknn = pickle.load(open('heart-disease-prediction-knn-model.pkl', 'rb'))
# Create application
app = Flask(__name__)

# Bind home function to URL

from flask import Flask,render_template,url_for,request
from flask_sqlalchemy import SQLAlchemy
from flask_material import Material
from flask import Flask
from flask_mail import Mail, Message
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
#from twilio.rest import Client
import math, random 
import requests
import pandas as pd 
import numpy as np 
import sqlite3
#from sqlite3 import Error
from flask import make_response
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
# ML Pkg
#from sklearn.externals 
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS']=True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'rajatmohantyrm768@gmail.com'
app.config['MAIL_PASSWORD'] = 'huibjwgevwwcaozp'
#app.config['MAIL_USE_TLS'] = False
#app.config['MAIL_USE_SSL'] = True
app.config['MAIL_MAX_EMAILS'] = None
mail = Mail(app)

Material(app)
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patient.SQLITE3'
#app.config['SECRET_KEY'] = "random string"
#db = SQLAlchemy(app)



@app.route('/')
def index():
    return render_template("home.html")

# Bind predict function to URL



@app.route('/showrf')
def showrf():
    return render_template('Diabetes.html')




@app.route('/predictrf', methods=['POST'])
def predictrf():
    allformvalues=list(request.form.values())
    print(allformvalues)
    pname=allformvalues[0]
    hid=allformvalues[1]
    allformvalues.remove(pname)
    allformvalues.remove(hid)
    print(allformvalues)
    # Put all form entries values in a list
    features = [float(i) for i in allformvalues]
    # Convert features to array
    print("FETURES SIZE",len(features))
    array_features = [np.array(features)]
    # Predict features
    
    dataset = pd.read_csv('diabetes_data_upload.csv')
    dataset.replace(to_replace=dict(No=0, Yes=1), inplace=True)
    dataset.replace(to_replace=dict(Negative=0, Positive=1), inplace=True)
    dataset = dataset.rename(columns={"Genital thrush": "Genital_thrush"})
    from sklearn.preprocessing import MinMaxScaler
    minmax = MinMaxScaler()
    dataset[['Age']] = minmax.fit_transform(dataset[['Age']])
    dataset[['Age']] = minmax.transform(dataset[['Age']])
    data = pd.read_csv('diabetes_dataset__2019.csv')
    data['RegularMedicine'].replace('o','no', inplace=True)
    data['BPLevel'] = data['BPLevel'].str.lower().str.strip()
    data['Pdiabetes'].replace('0', 'no', inplace=True)
    data['Diabetic'] = data['Diabetic'].str.strip()
    # there is nan value at pregancies column where gender is male 
    # if these values are replaced with 0, there's only 26 values, so all nan values will be replaced with 0. 
    data[data['Gender']=='Male']['Pregancies'].isna().sum()
    data['Pregancies'].replace(np.nan, 0, inplace=True)
# will drop all na's 
    data.dropna(inplace=True)
    data.info()

    num_cols = ['BMI', 'Sleep', 'SoundSleep', 'Pregancies']
    category_cols = list(set(data.columns).difference(set(num_cols)))

    data_clean = pd.DataFrame()
    for col in num_cols: 
        data_clean[col] = data[col].astype('int')
    for col in category_cols:
        data_clean[col] = data[col].astype('category')

    data_clean['Age'] = pd.Categorical(data['Age'], ordered=True, 
                                   categories=['less than 40', '40-49', '50-59', '60 or older'])
    data_clean['PhysicallyActive'] = pd.Categorical(data['PhysicallyActive'], ordered=True, 
                                                    categories=['one hr or more', 'more than half an hr', 'less than half an hr', 'none'])
    data_clean['JunkFood'] = pd.Categorical(data['JunkFood'], ordered=True, categories=['occasionally', 'often', 'very often', 'always'])
    data_clean['BPLevel'] = pd.Categorical(data['BPLevel'], ordered=True, 
                                        categories=['low', 'normal', 'high'])
    data_clean['Stress'] = pd.Categorical(data['Stress'], ordered=True, 
                                        categories=['not at all', 'sometimes', 'very often', 'always'])
    

    category_mapping = {
    'Age':{'less than 40':0, '40-49':1, '50-59':2, '60 or older':3},
    'Family_Diabetes':{'no':0, 'yes':1},
    'Gender':{'Female':0, 'Male':1},
    'Smoking':{'no':0, 'yes':1},
    'Pdiabetes':{'no':0, 'yes':1},
    'RegularMedicine':{'no':0, 'yes':1},
    'PhysicallyActive':{'one hr or more':0, 'more than half an hr':1, 'less than half an hr':2, 'none':3},
    'JunkFood':{'occasionally':0, 'often':1, 'very often':2, 'always':3},
    'BPLevel':{'low':0, 'normal':1, 'high':2},
    'highBP':{'no':0, 'yes':1},
    'Alcohol':{'no':0, 'yes':1},
    'UriationFreq':{'not much':0, 'quite often':1},
    'Stress':{'not at all':0, 'sometimes':1, 'very often':2, 'always':3},
    'Diabetic':{'no':0, 'yes':1},
}

    for col in category_cols:
        data_clean[col] = data_clean[col].map(category_mapping[col])

    data_2 = data_clean
    print("DATA2 ",data_2)
    data_2=data_clean.rename(columns={"Diabetic":"class"})
    df = pd.merge(dataset, data_2,on='class')

    last_column = df.pop('class')
  
# insert column using insert(position,column_name, first_column) function
    df.insert(33, 'class', last_column)
    df=df[df.Age_x<=80] [df.Genital_thrush<=2] [df.Irritability<=2] [df.Obesity<=2] [df.BMI<=38] [df.Pregancies<=1]
    print(df)
    #df=df['Gender_x'].replace("Female",0,inplace=True)
    #df=df['Gender_x'].replace("Male",1,inplace=True)
    df=df.drop(['Gender_x','Age_y'], axis=1)
    #X = df.iloc[:, 32].values
    X = df.iloc[:, 0:31].values
    print("COLUMN LIST   ",list(df.columns))
    print("FEATURES  ",list(df.iloc[:,0:31]))
    X=np.asarray(X).astype(np.float32)
    print("VALUE OF X  ",X)
    #y = df.iloc[:, 31].values
    y=df['class']
    print("TARGET  ",df['class'])
    y=np.asarray(y).astype(np.float32)

    print("VALUE OF Y ",y)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    print("X TRAIN VALUE ",X_train)
    print("X TEST VALUE",X_test)

    print("Y TRAIN VALUE",y_train)
    print("Y TEST VALUE",y_test)


    rf_grid = {"n_estimators": np.arange(10, 1000, 50),
          "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}

    model = RandomForestClassifier() # , max_depth=5, random_state=1
    np.random.seed(42)
    rs_rf=RandomizedSearchCV(RandomForestClassifier(),param_distributions=rf_grid,cv=5,n_iter=20,verbose=True)
    model.fit(X_train, y_train)
    model.fit(X_train, y_train)
    Y_pred = model.predict(X_test)
    score = model.score(X_train, y_train)
    t1score=score
    print('Training Score:', score)
    score = model.score(X_test, y_test)
    t2score=round(score)
    print('Testing Score:', score)
    output = pd.DataFrame({'Predicted':Y_pred}) # Heart-Disease yes or no? 1/0
    print(output.head())
    people = output.loc[output.Predicted == 1]["Predicted"]
    rate_people = 0
    if len(people) > 0 :
        rate_people = len(people)/len(output)
    print("% of people predicted with heart-disease:", rate_people)
    score_rfc = score
    out_rfc = output
    
    print(classification_report(y_test,Y_pred))
    prediction = model.predict(array_features)

    output = prediction
    print("OUTPUT VALUE",output)
    x = model.predict_proba(array_features)
    pos = x[0][1]
    pos = pos*100

    # neg = x[0][0]

    output = prediction

    if pos > 70:
        pred="Risk is high"
    if pos > 40:
        pred="Risk is medium"
    else:
        pred="Risk is low"
    

    if pos > 70:
        return render_template('Diabetes.html',
                               result='Probablity of having Diebetes disease: ', positive=pos, res2='Risk is HIGH',score=score_rfc)
    if pos > 40:
        return render_template('Diabetes.html',
                               result='Probablity of having Diebetes disease: ', positive=pos, res2='Risk is MEDIUM',score=score_rfc)
    else:
        return render_template('Diabetes.html',
                               result='Probablity of having Diebetes disease: ', positive=pos, res2='Risk is LOW',score=score_rfc)





    #from keras.models import Sequential
    #from keras.layers import Dense

    #model = Sequential()
    #model.add(Dense(input_dim = 1, units = 10, activation='relu', kernel_initializer='uniform'))
    #model.add(Dense(units = 10, activation='relu', kernel_initializer='uniform'))

    #model.add(Dense(units = 1, activation='sigmoid', kernel_initializer='uniform'))

    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #history=model.fit(X_train,y_train,batch_size=50, epochs=20, validation_split=0.2)

    #y_pred = model.predict(X_test)
    #print("XTEST VALUE  ",X_test)
    #print("XTEST VALUE FIRST ROW  ",X_test[0])
    #y_pred = (y_pred > 0.5)

    #y_pred2 = model.predict()

    #print("VALUE OF Y-PRED  ",y_pred)
    #print("VALUE OF Y-PRED FIRST ROW ",y_pred[0])

    #y_pred2 = model.predict([22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    #print("Y PRED2 ",y_pred2)






    




if __name__ == '__main__':
    # Run the application
    app.run(debug=True)
