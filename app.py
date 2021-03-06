# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:14:57 2022

@author: Sumit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.io.wavfile import write
from sklearn.metrics import mean_squared_error
import math

st.set_page_config(layout='wide',page_title="Music")
st.set_option('deprecation.showPyplotGlobalUse', False)


def convert_df(df):
    return df.to_csv().encode('utf-8')


html_header='''
<div style= 'background-color: pink; padding:13px';>
<h2 style= "color:black; text-align:center;"><b> Music- Best Curve fit</b></h2>
</div>
<br>
'''

st.markdown(html_header, unsafe_allow_html=True)


df = pd.read_csv('sample.csv')
df_original= df.copy()
x= df['Time'].values.tolist()
y= df['Amplitude'].values.tolist()


col1, col2= st.columns(2)

time_frame = col1.slider( 'Select time frame', min(x), max(x), (min(x) + 0.25, min(x) + 0.75))
#time_frame = st.slider( 'Select time frame', 0, len(df), ( 0, len(df)))
print(time_frame)

df= df[df['Time']>= time_frame[0] ]
df= df[df['Time']<= time_frame[1]]


#df= df[time_frame[0]: time_frame[1]]
x= df['Time'].values.tolist()
y= df['Amplitude'].values.tolist()


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range = (1,100))
#df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


df= df.set_index('Time')
#df_scaled= df_scaled.set_index('Time')






#df = df.rename(columns={'Time':'index'}).set_index('index')

#st.write(df)

#fig = plt.figure(figsize=(5, 3))
#plt.scatter(x, y)
col1.line_chart(df)
#col1.line_chart(df_scaled)

#st.pyplot(fig)

#list_of_functions=['M-M kinetics', 'L-W plot', 'Bateman function', 'Nuclear decay', 'Chemical reactions', 'Sinusodial functions', 'Polynomial functions', 'Logarithmic & Exponential', 'Continuous fibonacci', 'Logistic growth', 'Probability distribution']
list_of_functions=[ 'Sinusodial functions', 'Sine waves','Polynomial functions']


st.sidebar.markdown("### Select function")
filter_function=[]
select_all = st.sidebar.checkbox("Select All", True)
if select_all:
    for i_func in list_of_functions:
        if st.sidebar.checkbox(i_func, True):
            filter_function.append(i_func)

else:
    for i_func in list_of_functions:
        if st.sidebar.checkbox(i_func, False):
            filter_function.append(i_func)



if len(filter_function) ==0:
    st.warning("Please select at least one Function")
 



else:
    if 'Polynomial functions' in filter_function:

        poly_n= st.sidebar.selectbox(" Select value of n",[1,2,3,4,5,6,7,8,9], 3)
        model = np.poly1d(np.polyfit(df.index, df.Amplitude, poly_n))
        
        
        polyline=df.index.values.tolist()
        amplitude_fit= model(polyline)
        fig = plt.figure(figsize=(4, 2))
        plt.scatter(df.index, df.Amplitude, c='k', label='Amplitude')
        plt.plot(polyline, model(polyline), color='red', label='Best fit')
        plt.legend(fontsize=4)
        col1.pyplot(fig)
        
        
        model= list(model)
        model= [round(num, 2) for num in model]
        if poly_n== 1:
            equation = '''y = (({} x) + ({})'''.format(model[0], model[1])
        elif poly_n== 2:
            equation = '''y = (({} x^2)  + ({} x) + ({})'''.format(model[0], model[1], model[2])
        elif poly_n== 3:
            equation = '''y = (({} x^3) + ({} x^2)  + ({} x) + ({})'''.format(model[0], model[1], model[2], model[3])
        elif poly_n== 4:
            equation = '''y = (({} x^4)  +({} x^3) + ({} x^2)  + ({} x) + ({})'''.format(model[0], model[1], model[2], model[3], model[4])
        elif poly_n== 5:
            equation = '''y = ( ({} x^5) + ({} x^4)  +({} x^3) + ({} x^2)  + ({} x) + ({})'''.format(model[0], model[1], model[2], model[3], model[4], model[5])
        elif poly_n== 6:
            equation = '''y = ( ( {} x^6) +   {} x^5) + ({} x^4)  +({} x^3) + ({} x^2)  + ({} x) + ({})'''.format(model[0], model[1], model[2], model[3], model[4], model[5], model[6])
        elif poly_n== 7:
            equation = '''y = ( (  {} x^7) +  {} x^6) +   {} x^5) + ({} x^4)  +({} x^3) + ({} x^2)  + ({} x) + ({})'''.format(model[0], model[1], model[2], model[3], model[4], model[5], model[6], model[7])
        elif poly_n== 8:
            equation = '''y = ( ( {} x^8) +  {} x^7) +  {} x^6) +   {} x^5) + ({} x^4)  +({} x^3) + ({} x^2)  + ({} x) + ({})'''.format(model[0], model[1], model[2], model[3], model[4], model[5], model[6], model[7], model[8])
        elif poly_n== 9:
            equation = '''y = ( ( {} x^9) + {} x^8) +  {} x^7) +  {} x^6) +   {} x^5) + ({} x^4)  +({} x^3) + ({} x^2)  + ({} x) + ({})'''.format(model[0], model[1], model[2], model[3], model[4], model[5], model[6], model[7], model[8], model[9])
        
        
        st.latex(equation)
        
        
        
        rmse_list=[]
        ploy_list=[]
        for i_poly in range(1,10):
            model = np.poly1d(np.polyfit(df.index, df.Amplitude, i_poly))
            polyline=df.index.values.tolist()
            amplitude_fit= model(polyline)
            
            df_fit= pd.DataFrame(columns= ['Time', 'Amplitude_fit', 'Amplitude'])
            df_fit['Time']= polyline
            df_fit['Amplitude_fit']= amplitude_fit
            df_fit['Amplitude']= df['Amplitude'].values.tolist()
            ploy_list.append(i_poly)
            rmse_list.append(mean_squared_error(df_fit[['Amplitude']], df_fit[['Amplitude_fit']], squared=False))
            
            
        df_rmse= pd.DataFrame(columns= ['Poly n', 'RMSE'])
        df_rmse['Poly n']= ploy_list
        df_rmse['RMSE']= rmse_list
        
        best_rmse= df_rmse['RMSE'].min()
        best_poly= df_rmse[df_rmse['RMSE']== best_rmse]['Poly n'].values.tolist()[0]
        
        col2.table(df_rmse)
        col2.success( "Best polynomial fit: " +  str(best_poly))
        
        model = np.poly1d(np.polyfit(df.index, df.Amplitude, best_poly))
        polyline=df.index.values.tolist()
        amplitude_fit= model(polyline)
        df_fit= pd.DataFrame(columns= ['Time', 'Amplitude_fit', 'Amplitude'])
        df_fit['Time']= polyline
        df_fit['Amplitude_fit']= amplitude_fit
        df_fit['Amplitude']= df['Amplitude'].values.tolist()
        
        df_result= df_fit.copy()
        
        df_fit = convert_df(df_fit)
        col2.download_button(
               "Download file",
               df_fit,
               "Curve_fit.csv",
               "text/csv",
               key='download-csv'
            )
        
        
        
        #df_original['Frequency_fit']= df_original['Amplitude']
        
        #st.write(df_original)
        df_result= df_result.drop('Amplitude', axis=1)
        df_result=  pd.merge(df_original, df_result, how='left', on=['Time'])
        df_result['Amplitude_fit'].fillna(df_result.Amplitude, inplace=True)
        #st.write(df_result)
        
        
        
        
        
        amplitude = 4096*2 #arbitrary value
        duration=0.01161 #this is from the data: sample_data['Time'][5]-sample_data['Time'][4]
        samplerate = 44100
        original_freq= df_original['Amplitude'].values.tolist()
        fit_freq= df_result['Amplitude_fit'].values.tolist()
        
        song=[]
        p = 0
        for note in original_freq:
            t = np.linspace(0, duration, int(samplerate * duration))
            wave = amplitude * np.sin(2 * np.pi * note * t + p) # seems like we can add our sample frequency here to get the wave
            song.append(wave)
            p = np.mod(2*np.pi*note*duration + p,2*np.pi) #to make sure that the next wave starts with the same phase where the previous wave ended

        song = np.concatenate(song) 
        data = song.astype(np.int16)
        data = data * (16300/np.max(data))
        write('sample_original.wav', samplerate, data.astype(np.int16))
        
        
        
        song=[]
        p = 0
        for note in fit_freq:
            t = np.linspace(0, duration, int(samplerate * duration))
            wave = amplitude * np.sin(2 * np.pi * note * t + p) # seems like we can add our sample frequency here to get the wave
            song.append(wave)
            p = np.mod(2*np.pi*note*duration + p,2*np.pi) #to make sure that the next wave starts with the same phase where the previous wave ended

        song = np.concatenate(song) 
        #song=np.array([x*2 for x in song]) 
        data = song.astype(np.int16)
        data = data * (16300/np.max(data))
        write('sample_fit.wav', samplerate, data.astype(np.int16))
        
        audio_file = open('sample_original.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
        
        audio_file_fit = open('sample_fit.wav', 'rb')
        audio_bytes_fit = audio_file_fit.read()
        st.audio(audio_bytes_fit, format='audio/wav')
            


    elif 'Sinusodial functions' in filter_function:
        
        # Test function with coefficients as parameters
        def sine_wave(x, a, b):
            return a * np.sin(b * x)
        
        def cos_wave(x, a, b):
            return a * np.cos(b * x)
        
        x= np.array(df.index.values.tolist())
        y= np.array(df['Amplitude'].values.tolist())
        
        
        # curve_fit() function takes the test-function
        # x-data and y-data as argument and returns
        # the coefficients a and b in param and
        # the estimated covariance of param in param_cov
        param_sin, param_cov = curve_fit(sine_wave, x, y)
        param_cos, param_cov_cos = curve_fit(cos_wave, x, y)
        
        # ans stores the new y-data according to
        # the coefficients given by curve-fit() function
        ans_sin = (param_sin[0]*(np.sin(param_sin[1]*x)))
        ans_cos= (param_cos[0]*(np.cos(param_cos[1]*x)))
        

        
        fig = plt.figure(figsize=(4, 2))
        #plt.plot(x, y, 'o', color ='red', label ="data")
        plt.scatter(x, y, c='k', label='Amplitude')
        plt.plot(x, ans_sin, '--', color ='blue', label ="Sine wave")
        plt.plot(x, ans_cos, '--', color ='green', label ="Cosine wave")
        plt.legend(fontsize=4)
        col1.pyplot(fig)
        
        
        
        df_fit= df.copy()
        df_fit['Sine wave']= ans_sin
        df_fit['Cosine wave']= ans_cos
        #col2.write(df_fit)
        rmse_sin= mean_squared_error(df_fit[['Amplitude']], df_fit[['Sine wave']], squared=False)
        rmse_cos= mean_squared_error(df_fit[['Amplitude']], df_fit[['Cosine wave']], squared=False)
        
        df_rmse= pd.DataFrame(columns= ['Wave equation', 'RMSE'])
        df_rmse['Wave equation']= ['Sine wave', 'Cosine wave']
        df_rmse['RMSE']= [rmse_sin, rmse_cos]
        
        best_rmse= df_rmse['RMSE'].min()
        best_wave= df_rmse[df_rmse['RMSE']== best_rmse]['Wave equation'].values.tolist()[0]
        
        col2.table(df_rmse)
        col2.success( "Best Wave equation: " +  str(best_wave))
            
        
        
        
        param1_sin= round(param_sin[0],2)
        param2_sin= round(param_sin[1],2)
        param1_cos= round(param_cos[0],2)
        param2_cos= round(param_cos[1],2)
        
        sin_equation= '''y= {} sin({} x)   '''.format(param1_sin, param2_sin)
        cos_equation= '''y= {} cos({} x)   '''.format(param1_cos, param2_cos)
        
        if best_wave== 'Sine wave':
            col2.latex( sin_equation)
        else:
            col2.latex( cos_equation)
            
            
        
        
        df_fit = convert_df(df_fit)
        col2.download_button(
               "Download file",
               df_fit,
               "Curve_fit.csv",
               "text/csv",
               key='download-csv'
            )
        




    elif 'Sine waves' in filter_function:

        df= df.reset_index()
        
        ym= round(df['Amplitude'].mean(), 2)
        xm= round(df['Time'].mean(), 2)
        
        y2= round(df['Amplitude'].iloc[-1], 2)
        x2= round(df['Time'].iloc[-1], 2)
        
        y1= round(df['Amplitude'].iloc[0], 2)
        x1= round(df['Time'].iloc[-0], 2)
        
        
        
        
        x= np.array(df['Time'].values.tolist())
        y= np.array(df['Amplitude'].values.tolist())
        y_sin=[]
        y_cos=[]
        
        for ix in x:
            y_sin.append( math.sin(((ix - xm)*(3.14/2)/2)*(1/(x2-xm))) *(y2-ym) + ym) 
            y_cos.append( math.sin(((ix - x1)*(3.14/2)/2)*(1/(x2-x1))) *(y2-y1) + y1)
            
        fig = plt.figure(figsize=(4, 2))
        plt.scatter(x, y, c='k', label='Amplitude')
        plt.plot(x, y_sin, '--', color ='blue', label ="Sine wave")
        plt.plot(x, y_cos, '--', color ='green', label ="Cosine wave")
        plt.legend(fontsize=4)
        col1.pyplot(fig)
        
        
        df_fit= df.copy()
        df_fit['Sine wave']= y_sin
        df_fit['Cosine wave']= y_cos
        #col2.write(df_fit)
        rmse_sin= mean_squared_error(df_fit[['Amplitude']], df_fit[['Sine wave']], squared=False)
        rmse_cos= mean_squared_error(df_fit[['Amplitude']], df_fit[['Cosine wave']], squared=False)
        
        
        df_rmse= pd.DataFrame(columns= ['Wave equation', 'RMSE'])
        df_rmse['Wave equation']= ['Sine wave', 'Cosine wave']
        df_rmse['RMSE']= [rmse_sin, rmse_cos]
        
        best_rmse= df_rmse['RMSE'].min()
        best_wave= df_rmse[df_rmse['RMSE']== best_rmse]['Wave equation'].values.tolist()[0]
        
        col2.table(df_rmse)
        col2.success( "Best Wave equation: " +  str(best_wave))
        
        sine_eq= '''y= sin[(x- {}) (3.14/2) ( 1/({}-{}))]({}-{}) + {}  '''.format(xm, x2, xm, y2, ym, ym)
        #col2.latex(sine_eq)
        if best_wave== 'Sine wave':
            col2.latex( sine_eq)
        else:
            col2.latex( sine_eq)
        
        

