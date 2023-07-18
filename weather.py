import tensorflow as tf
import os

import streamlit as st
import datetime,requests
from plotly import graph_objects as go
from inference import predicting,get_n_latest
import pandas as pd



province_list=["Bac Lieu", "Ben Tre", "Bien Hoa", "Buon Me Thuot", "Ca Mau", "Cam Pha", "Cam Ranh", "Can Tho", "Chau Doc", "Da Lat", "Ha Noi", "Hai Duong", "Hai Phong", "Hanoi", "Ho Chi Minh City", "Hoa Binh", "Hong Gai", "Hue", "Long Xuyen", "My Tho", "Nam Dinh", "Nha Trang", "Phan Rang", "Phan Thiet", "Play Cu", "Qui Nhon", "Rach Gia", "Soc Trang", "Tam Ky", "Tan An", "Thai Nguyen", "Thanh Hoa", "Tra Vinh", "Tuy Hoa", "Uong Bi", "Viet Tri", "Vinh", "Vinh Long", "Vung Tau", "Yen Bai"]


st.set_page_config(page_title='Vietnamese_Wearher_forecast', page_icon=":rainbow:")

st.title("WEATHER FORECAST 🌧️🌥️")

st.markdown("[Dataset record from 1st Jan 2009 to Jun 18th 2021 of 40 province or city in Vietnam](https://www.kaggle.com/datasets/vanviethieuanh/vietnam-weather-data?fbclid=IwAR3wCC9HFy09sjtiGcClHPIrnA7MrhJpzSMaQ_-kuC-pFee8JgJindOuT_M)")

province=st.selectbox("SELECT THE VIETNAMESE'S PROVINCE",province_list)

forecast_day=st.selectbox("SELECT THE NUMBER OF DAY FORECAST",[8,7,6,5,4,3,2,1])

unit=st.selectbox("SELECT TEMPERATURE UNIT ",["Celsius","Fahrenheit"])

speed=st.selectbox("SELECT WIND SPEED UNIT ",["Metre/sec","Kilometre/hour"])

graph=st.radio("SELECT GRAPH TYPE ",["Bar Graph","Line Graph"])

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://wallpaperaccess.com/full/1442216.jpg")
    }
  
    </style>
    """,
    unsafe_allow_html=True
)

if "submitted" not in st.session_state:
    st.session_state.submitted = False

def submitted():
    st.session_state.submitted = True
def reset():
    st.session_state.submitted = False


col1, col2= st.columns([.15,1])
with col1:
    update= st.button("UPDATE",on_click=reset)
with col2:
    forecast= st.button("FORECAST",on_click=reset)


if unit=="Celsius":
    temp_unit=" °C"
else:
    temp_unit=" °F"
    
if speed=="Kilometre/hour":
    wind_unit=" km/h"
else:
    wind_unit=" m/s"

df_path = './DBM301/clean_data.csv'
df = pd.read_csv(df_path, index_col=[0])
df['date']=pd.to_datetime(df['date'])




if(update):
    lastest= get_n_latest(df,province, number_of_days=10)

    maxtemp=[]
    mintemp=[]
    pres=[]
    humd=[]
    wspeed=[]
    wdir=[]
    cloud=[]
    rain=[]
    dates=[]

    for index, row in lastest.iterrows():
        
        if unit=="Celsius":
            maxtemp.append(round(row["max"],2))
            mintemp.append(round(row["min"],2))
        else:
            maxtemp.append(round(((row["max"]*1.8)+32),2))
            mintemp.append(round(((row["min"]*1.8)+32),2))

        if wind_unit=="m/s":
            wspeed.append(str(round(row["wind"]/3.6,1))+wind_unit)
        else:
            wspeed.append(str(round(row["wind"],1))+wind_unit)

        wdir.append(row['wind_d'])

        pres.append(row["pressure"])
        humd.append(str(row["humidi"])+' %')
        
        cloud.append(str(row["cloud"])+' %')
        rain.append(str(row["rain"])+'mm')

        d1=row["date"]
        dates.append(d1.strftime('%d %b'))


    ## display

    current= lastest['date'].iloc[0].strftime('%Y-%m-%d')
    onest= lastest['date'].iloc[-1].strftime('%Y-%m-%d')


    st.write(f"## Weather from {onest} - {current}")

    st.subheader(" ")
    table1=go.Figure(data=[go.Table(header=dict(
                values = [
                '<b>DATES</b>',
                '<b>MAX TEMP<br>(in'+temp_unit+')</b>',
                '<b>MIN TEMP<br>(in'+temp_unit+')</b>',
                '<b>RAIN</b>',
                '<b>CLOUD COVERAGE</b>',
                '<b>HUMIDITY</b>',
                '<b>WIND SPEED</b>',
                '<b>WIND DIRECTION</b>',
                '<b>PRESSURE<br>(in Pa)</b>'],
                line_color='black', fill_color='royalblue',  font=dict(color='white', size=14),height=32),
    
    cells=dict(values=[dates,maxtemp,mintemp,rain,cloud,humd,wspeed,wdir,pres],
    line_color='black',fill_color=['paleturquoise',['palegreen', '#fdbe72']*7], font_size=14,height=32))])

    table1.update_layout(margin=dict(l=10,r=10,b=10,t=10),height=328)
    st.write(table1)
    

    #Input new weather measure

    next_input_date = lastest['date'].iloc[-1] + pd.Timedelta(days=1)
    # print(type(next_input_date))

    with st.form("weather_update_form"):
        st.write(f"### Input Weather for {next_input_date.strftime('%Y-%m-%d')}")
        col1, col2= st.columns(2)
        with col1:
            max_in = st.number_input(label=f'Max ({temp_unit})',value=30.00, key="max")
            min_in = st.number_input(label=f'Min ({temp_unit})', value=20.00, key='min')
        with col2:
            wspeed_in = st.number_input(label=f'Wind speed ({wind_unit})', value=15.00, key='wspeed')
            wdir_in = st.selectbox("Wind direction",["N","NNE","NE","ENE","E","ESE","SE","SSE",
                                                    "S","SSW","SW","WSW","W","WNW","NW","NNW"],key="wdir")


        rain_in = st.number_input(label='Rain (mm)', value=0.00, key="rain")
        cloud_in = st.slider(label='Cloud (%)',value=50, min_value=0, max_value=100, key= 'cloud')
        humdi_in = st.slider(label='Humidity (%)', value=50, min_value=0, max_value=100, key= "humdi")
    
        pres_in = st.number_input(label='Pressure (in Pa)', value=1000.00, key= "pres")
        st.session_state.province= province
        st.session_state.date= next_input_date

        if unit=="Fahrenheit":#convert to celsius
            st.session_state.max=round(((max_in-32)*(5/9)),2)
            st.session_state.min=round(((max_in-32)*(5/9)),2)


        # Every form must have a submit button.
        st.form_submit_button("Submit", on_click=submitted)

def check_valid(new_weather):
    valid=False
    message=[]
    if new_weather["max"]< new_weather["min"]:
        message.append("Input Error: min < max")
    elif new_weather["wind"] < 0:
        message.append("Input Error: wind < 0")
    elif new_weather["rain"] < 0:
        message.append("Input Error: rain < 0")
    elif new_weather["pressure"] < 0:
        message.append("Input Error: pressure < 0")
    else:
        message.append('New weather updated successfully!')
        valid= True
    return valid,message
    


if st.session_state.submitted:
    new_weather={
        "province":st.session_state.province,
        "max":st.session_state.max,
        "min":st.session_state.min,
        "wind":st.session_state.wspeed,
        "wind_d":st.session_state.wdir,
        "rain":st.session_state.rain,
        "humidi":st.session_state.humdi,
        "cloud":st.session_state.cloud,
        "pressure":st.session_state.pres,
        "date":st.session_state.date,
    }
    st.write(new_weather)
    valid, message = check_valid(new_weather)
    if valid:
        new_weather=pd.DataFrame.from_records([new_weather])
        df = pd.concat([df,new_weather], ignore_index = True)
        df.to_csv("./DBM301/clean_data.csv") 
        st.success("\n-".join(message))
    else:
        st.error("\n-".join(message))
    



    
if(forecast):
    # try:
        #load model
        file_path = './DBM301/model2.h5'

        if os.path.exists(file_path):
            model = tf.keras.models.load_model(file_path)
        else:
            print("Loading Failed!")
            exit()

        model.summary()

        all=predicting(df, province, forecast_day, model)
        predict=all[-forecast_day:]
        current_weather=all[-(forecast_day+1):-forecast_day]

        maxtemp=[]
        mintemp=[]
        pres=[]
        humd=[]
        wspeed=[]
        cloud=[]
        rain=[]
        dates=[]

        for index, row in predict.iterrows():
            
            if unit=="Celsius":
                maxtemp.append(round(row["max"],2))
                mintemp.append(round(row["min"],2))
            else:
                maxtemp.append(round(((row["max"]*1.8)+32),2))
                mintemp.append(round(((row["min"]*1.8)+32),2))

            if wind_unit=="m/s":
                wspeed.append(str(round(row["wind"]/3.6,1))+wind_unit)
            else:
                wspeed.append(str(round(row["wind"],1))+wind_unit)

            pres.append(row["pressure"])
            humd.append(str(row["humidi"])+' %')
            
            cloud.append(str(row["cloud"])+' %')
            rain.append(str(row["rain"])+'mm')

            d1=index
            dates.append(d1.strftime('%d %b'))
            
            
        def bargraph():
            fig=go.Figure(data=
                [
                go.Bar(name="Maximum",x=dates,y=maxtemp,marker_color='crimson'),
                go.Bar(name="Minimum",x=dates,y=mintemp,marker_color='navy')
                ])
            fig.update_layout(xaxis_title="Dates",yaxis_title="Temperature",barmode='group',margin=dict(l=70, r=10, t=80, b=80),font=dict(color="white"))
            st.plotly_chart(fig)
        
        def linegraph():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=mintemp, name='Minimum '))
            fig.add_trace(go.Scatter(x=dates, y=maxtemp, name='Maximimum ',marker_color='crimson'))
            fig.update_layout(xaxis_title="Dates",yaxis_title="Temperature",font=dict(color="white"))
            st.plotly_chart(fig)
            
        
        if unit=="Celsius":
            cur_min=str(round(current_weather["min"][0],2))
            cur_max=str(round(current_weather["max"][0],2))
        else:
            cur_min=str(round(((current_weather["min"][0]*1.8)+32),2))
            cur_max=str(round(((current_weather["max"][0]*1.8)+32),2))
        
        d= pd.to_datetime(str(current_weather.index.values[0]))
        cur_date= d.strftime('%Y-%m-%d')

        st.write(f"## Last Update: {cur_date}")
        
        
        col1, col2= st.columns(2)
        col1.metric("MAX TEMPERATURE", cur_max+temp_unit)
        col2.metric("MIN TEMPERATURE", cur_min+temp_unit)
        st.subheader(" ")
        
        if graph=="Bar Graph":
            bargraph()
            
        elif graph=="Line Graph":
            linegraph()

         
        table1=go.Figure(data=[go.Table(header=dict(
                  values = [
                  '<b>DATES</b>',
                  '<b>MAX TEMP<br>(in'+temp_unit+')</b>',
                  '<b>MIN TEMP<br>(in'+temp_unit+')</b>',
                  '<b>RAIN</b>',
                  '<b>CLOUD COVERAGE</b>',
                  '<b>HUMIDITY</b>',
                  '<b>WIND SPEED</b>','<b>PRESSURE<br>(in Pa)</b>'],
                  line_color='black', fill_color='royalblue',  font=dict(color='white', size=14),height=32),
        
        cells=dict(values=[dates,maxtemp,mintemp,rain,cloud,humd,wspeed,pres],
        line_color='black',fill_color=['paleturquoise',['palegreen', '#fdbe72']*7], font_size=14,height=32))])

        table1.update_layout(margin=dict(l=10,r=10,b=10,t=10),height=328)
        st.write(table1)
        
        # table2=go.Figure(data=[go.Table(columnwidth=[1,2,1,1,1,1],header=dict(values=['<b>DATES</b>','<b>WEATHER CONDITION</b>','<b>WIND SPEED</b>','<b>PRESSURE<br>(in hPa)</b>','<b>SUNRISE<br>(in UTC)</b>','<b>SUNSET<br>(in UTC)</b>']
        #           ,line_color='black', fill_color='royalblue',  font=dict(color='white', size=14),height=36),
        # cells=dict(values=[dates,wspeed,pres,sunrise,sunset],
        # line_color='black',fill_color=['paleturquoise',['palegreen', '#fdbe72']*7], font_size=14,height=36))])
        
        # table2.update_layout(margin=dict(l=10,r=10,b=10,t=10),height=360)
        # st.write(table2)
        
        # st.header(' ')
        # st.header(' ')
        # st.markdown("Made with :heart: by : ")
        # st.markdown("Nikhilesh Shah 🤩 && Nandita Agarwal 🤗 && Nisha Vaghela 🥰")
 
    # except KeyError:
    #     st.error(" Invalid city!!  Please try again !!")

