    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib
    d="2020-10-12"
    ws=[]
    wd=[]
    m=[]
    for x in range(144):
        ws.append(np.random.uniform(0,25.5))
        wd.append(np.random.uniform(0,360))
        m.append(d)    
        
    data=np.concatenate([np.array(ws).reshape(144,1),np.array(wd).reshape(144,1),np.array(m).reshape(144,1)],1)
    df1=pd.DataFrame(data=data,columns=["Wind Speed","Wind direction","Date/Time"])

    x=0
    y=0
    def date_time(t):
            global x
            global y
            t=t + " "+ str(y)+":"+str(x*10)+":"+"00" # this is where problem occur
            if x<5:
                x=x+1
            else:
                x=0
                y=y+1
            return t    
    df1["Date/Time"]=df1["Date/Time"].apply(date_time,1)            

    df1["Date/Time"]=pd.to_datetime(df1["Date/Time"])
    df1=df1.set_index("Date/Time")
    df1["Year"]=df1.index.year
    df1["Month"]=df1.index.month
    df1["Weekday"]=df1.index.weekday

    X1=df1[["Wind Speed","Wind direction"]]
    from sklearn.preprocessing import StandardScaler
    scaler1=StandardScaler()
    scaler1.fit(X1)
    scaled_data1=scaler1.transform(X1)

    df1["Theoretical_Power_Curve (KWh)"]=joblib.load('Wind Turbine(TPC).sav').predict(scaled_data1)

    X2=df1[["Wind Speed","Theoretical_Power_Curve (KWh)","Month"]]
    from sklearn.preprocessing import StandardScaler
    scaler2=StandardScaler()
    scaler2.fit(X2)
    scaled_data2=scaler2.transform(X2)

    df1["LV ActivePower (kW)"]=joblib.load('Wind Turbine(LV).sav').predict(scaled_data2)


fig, axes = plt.subplots(figsize=(12,4))
x=df1.loc[d].index
y=df1["LV ActivePower (kW)"].values
z=df1["Theoretical_Power_Curve (KWh)"].values
axes.plot(x, y, 'b')
axes.plot(x,z,"g")
axes.set_ylabel('LV ActivePower (kW)')
axes.set_xlabel('Date/Time')
axes.set_title('Time Series Analysis')
plt.show()
plt.savefig("Time Series Analysis.png")
    
    