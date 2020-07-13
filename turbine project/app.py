import numpy as np
from flask import Flask, request, render_template, url_for
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt


GRAPH_FOLDER= os.path.join('static')
app = Flask(__name__)
app.config['upload']=GRAPH_FOLDER

x=0
y=0
def date_time(t):
            global x
            global y
            t=t + "  "+ str(y)+":"+str(x*10)+":"+"00" 
            if x<5:
                x=x+1
            else:
                x=0
                y=y+1
            return t    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    global d
    d=request.form['Date']
    d1=d[5:10]
    df=pd.read_csv("T1.csv")
    df["Date"]=df["Date/Time"].apply(lambda x:x[0:2]+"-"+x[3:5])
    m=[]
    for z in range(144):
          m.append(d)
    df1=pd.DataFrame(data=m,columns=["Date/Time"])
    df1["Date"]=df1["Date/Time"].apply(lambda i:i[5:10])
    df=df.set_index("Date")
    df1=df1.set_index("Date")
    df1=pd.concat([df1,df.loc[d1][["Wind Speed (m/s)","Wind Direction (°)"]]],1)
    
    
    
    df1["Date/Time"]=df1["Date/Time"].apply(date_time,1)
    global x
    x=0
    global y
    y=0 
    df1["Time"]=df1["Date/Time"].apply(lambda x:x[-8:-3],1)          
    df1["Hour"]=pd.to_timedelta(df1["Date/Time"].apply(lambda x:x[-8:],1)).dt.components["hours"]
    df1[['date','time']] = df1['Date/Time'].str.split(expand=True)
    df1['Date/Time'] = (pd.to_datetime(df1.pop('date'), format='%Y/%m/%d') + 
                  pd.to_timedelta(df1.pop('time') ))
    df1=df1.set_index("Date/Time")
    df1["Year"]=df1.index.year
    df1["Month"]=df1.index.month
    df1["Weekday"]=df1.index.weekday

    X1=df1[["Wind Speed (m/s)","Wind Direction (°)"]]
    from sklearn.preprocessing import StandardScaler
    scaler1=StandardScaler()
    scaler1.fit(X1)
    scaled_data1=scaler1.transform(X1)

    df1["Theoretical_Power_Curve (KWh)"]=joblib.load('Wind Turbine(TPC).sav').predict(scaled_data1)

    X2=df1[["Wind Speed (m/s)","Wind Direction (°)","Month","Hour"]]
    from sklearn.preprocessing import StandardScaler
    scaler2=StandardScaler()
    scaler2.fit(X2)
    scaled_data2=scaler2.transform(X2)

    df1["LV ActivePower (kW)"]=joblib.load('Wind Turbine(LV).sav').predict(scaled_data2)

    prediction1=df1["LV ActivePower (kW)"].mean()
    prediction2=df1["Theoretical_Power_Curve (KWh)"].mean()
    fig, axes = plt.subplots(figsize=(80,20))
    x1=df1["Time"]
    y1=df1["LV ActivePower (kW)"].values
    z1=df1["Theoretical_Power_Curve (KWh)"].values
    axes.plot(x1, y1, 'b',label="LV ActivePower")
    axes.plot(x1,z1,"r",label="Theoretical_Power_Curve")
    axes.legend(fontsize=50,loc='upper right')
    plt.xlabel('Date/Time',fontsize=100)
    plt.ylabel('Power (kW)',fontsize=100)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=50)
    axes.set_xticklabels(x1,rotation=90)
    axes.set_title('Time Series Analysis')
    axes.title.set_size(100)
    plt.savefig("static/temp.png")
    

    if d[0:4]=="2018":
        return render_template('index.html', prediction_text=str(prediction1),theoretical=str(prediction2),status="Predicted",max=str(df1["LV ActivePower (kW)"].max()),min=str(df1["LV ActivePower (kW)"].min()))
    else:
        return render_template('index.html',prediction_text="None",theoretical="None",status="Sorry!,We have no any forecasted data for this date.",max="None",min="None")

@app.route('/display',  methods=["POST"])
def display():
    full_filename1= os.path.join(app.config['upload'], 'temp.png')
    full_filename2= os.path.join(app.config['upload'], 'try again.png')
    if d[0:4]=="2018":
        return render_template('graph.html', graph_image=full_filename1 )
    else:
        return render_template('graph.html', graph_image=full_filename2 )
    



if __name__ == "__main__":
    app.run(debug=True)