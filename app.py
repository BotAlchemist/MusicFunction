# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:14:57 2022

@author: Sumit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

st.set_page_config(layout='wide',page_title="Music")
st.set_option('deprecation.showPyplotGlobalUse', False)

html_header='''
<div style= 'background-color: pink; padding:13px';>
<h2 style= "color:black; text-align:center;"><b> Music- Best Curve fit</b></h2>
</div>
<br>
'''

st.markdown(html_header, unsafe_allow_html=True)
st.sidebar.selectbox('Choose function', ['Polynomial'])

df = pd.read_csv('sample.csv')
x= df['Time'].values.tolist()
y= df['Amplitude'].values.tolist()


col1, col2= st.beta_columns(2)

time_frame = col1.slider( 'Select time frame', min(x), max(x), (min(x) + 0.25, min(x) + 0.75))
#time_frame = st.slider( 'Select time frame', 0, len(df), ( 0, len(df)))
print(time_frame)

df= df[df['Time']>= time_frame[0] ]
df= df[df['Time']<= time_frame[1]]


#df= df[time_frame[0]: time_frame[1]]
x= df['Time'].values.tolist()
y= df['Amplitude'].values.tolist()

df= df.set_index('Time')



#df = df.rename(columns={'Time':'index'}).set_index('index')

#st.write(df)

#fig = plt.figure(figsize=(5, 3))
#plt.scatter(x, y)
col1.line_chart(df)

#st.pyplot(fig)


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


#st.latex(equation)



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


