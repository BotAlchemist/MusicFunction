# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:14:57 2022

@author: Sumit
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.io.wavfile import write
from scipy import signal
from sklearn.metrics import mean_squared_error
import math
import plotly.express as px

from Bezier import Bezier


st.set_page_config(layout='wide',page_title="Music")
st.set_option('deprecation.showPyplotGlobalUse', False)


def convert_df(df):
    return df.to_csv().encode('utf-8')


html_header='''
<div style= 'background-color: pink; padding:13px';>
<h2 style= "color:black; text-align:center;"><b> Gannygit</b></h2>
</div>
<br>
'''

#st.sidebar.markdown(html_header, unsafe_allow_html=True)


#i_page= st.sidebar.selectbox("Page", ['Curve fitting', 'Canvas', 'Generate Music'])

with st.sidebar:
    i_page= option_menu('Gannygit', ['Curve fitting', 'Canvas', 'Generate Music', 'Bezier Curve'],
                        default_index=0, icons=['gear', 'paperclip','music-note-list', 'bezier' ], menu_icon= 'cast')

if i_page == 'Curve fitting':
    df = pd.read_csv('sample.csv')
    df_original= df.copy()
    x= df['Time'].values.tolist()
    y= df['Amplitude'].values.tolist()
    
    time_frame = st.slider( 'Select time frame', min(x), max(x), (min(x) + 0.25, min(x) + 0.75))
    col1, col2= st.columns(2)
    
    
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
            
            
            #st.write( model(polyline))
            
            y_scaled=[]
            x_scaled=[]
            for x in range(len(amplitude_fit)):
                #st.write(x)
                #y= (({} x^4)  +({} x^3) + ({} x^2)  + ({} x) + ({})'''.format(model[0], model[1], model[2], model[3], model[4])
                y= (model[0] * x**4)  + (model[1] * x**3 ) + (model[2]* x**2) + (model[3] * x**1) + model[4]
                y_scaled.append(y)
                x_scaled.append(x)
                
            fig = plt.figure(figsize=(4, 2))
            #plt.scatter(x_scaled, df.Amplitude, c='k', label='Amplitude')
            y_scaled = [float(i)/max(model(polyline)) for i in model(polyline)]
            x_scaled= [float(i)/max(x_scaled) for i in x_scaled]
            plt.plot(x_scaled, y_scaled, color='red', label='Scaled')
            plt.legend(fontsize=4)
            col1.pyplot(fig)
            
            st.write(model(polyline))
                
            
                
    
    
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
            
            # ym= round(df['Amplitude'].mean(), 2)
            # xm= round(df['Time'].mean(), 2)
            
            y2= round(df['Amplitude'].iloc[-1], 2)
            x2= round(df['Time'].iloc[-1], 2)
            
            y1= round(df['Amplitude'].iloc[0], 2)
            x1= round(df['Time'].iloc[-0], 2)
            
            ym= round((y2+ y1)/ 2, 2)
            xm= round((x2+ x1)/ 2, 2)
            
            # st.write(x1, y1)
            # st.write(x2, y2)
            # st.write(xm, ym)
            
            
            
            
            x= np.array(df['Time'].values.tolist())
            y= np.array(df['Amplitude'].values.tolist())
            y_sin=[]
            y_cos=[]
            
            for ix in x:
                #y_sin.append( math.sin(((ix - xm)*(3.14/2)/2)*(1/(x2-xm))) *(y2-ym) + ym) 
                #y_cos.append( math.sin(((ix - x1)*(3.14/2)/2)*(1/(x2-x1))) *(y2-y1) + y1)
                
                y_sin_temp= math.sin( (ix- xm)* (3.14/2) *  (1/ (x2- xm))) 
                y_sin.append((y_sin_temp * (y2-ym)) + ym)
                
                #y_cos_temp= math.sin( (ix- xm)* (3.14/2) *  (1/ (x2- xm))) 
                #y_sin.append((y_sin_temp * (y2-ym)) + ym)
            
                
            #st.write(len(x), len(y_sin))
            fig = plt.figure(figsize=(4, 2))
            plt.scatter(x, y, c='k', label='Amplitude')
            plt.plot(x, y_sin, '--', color ='blue', label ="Sine wave")
            #plt.plot(x, y_cos, '--', color ='green', label ="Cosine wave")
            plt.legend(fontsize=4)
            col1.pyplot(fig)
            
            
            df_fit= df.copy()
            df_fit['Sine wave']= y_sin
            #df_fit['Cosine wave']= y_cos
            #col2.write(df_fit)
            rmse_sin= mean_squared_error(df_fit[['Amplitude']], df_fit[['Sine wave']], squared=False)
            #rmse_cos= mean_squared_error(df_fit[['Amplitude']], df_fit[['Cosine wave']], squared=False)
            
            
            df_rmse= pd.DataFrame(columns= ['Wave equation', 'RMSE'])
            df_rmse['Wave equation']= ['Sine wave']
            df_rmse['RMSE']= [rmse_sin]
            
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
            
elif i_page == 'Canvas':
    drawing_mode = st.sidebar.selectbox(
    "Drawing tool:",
    ("freedraw", "line", "rect", "circle", "transform", "polygon", "point")
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3, key= 'stroke_slider')
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)

    
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    
    # Create a canvas component
    canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=150,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    display_toolbar=st.sidebar.checkbox("Display toolbar", True),
    key="full_app",
)
    
    
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects)

    
elif i_page == 'Generate Music':
    #http://www.soundofindia.com/showarticle.asp?in_article_id=-446619640 
    
    
    
    def get_note_freq(S_freq):
        octave= ['S','r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N' ,'S^']
        note_freq= {octave[i]: S_freq * pow (2, (i/12)) for i in range(len(octave))}
        note_freq['']= 0.0
        #note_freq['S^']= 2* S_freq
        
        return note_freq
    
    
    S_freq= st.sidebar.text_input("Select base frequency", '100')
    S_freq= float(S_freq)
    
    beats_per_note= st.sidebar.slider("Beats per note", 1,5)
    poly_n= st.sidebar.selectbox(" Smoothening level",[0,1,2,3,4,5,6,7,8,9], 0)
    
    notes_freq_dict= get_note_freq(S_freq)
    
    if 'user_input_sequence' not in st.session_state:
        st.session_state['user_input_sequence']= []
    
    mus1, mus2, mus3, mus4, mus5, mus6, mus7, mus8, mus9, mus10, mus11, mus12, mus13, mus14, mus15, mus16= st.columns(16)
    if mus1.button('स '):
        st.session_state['user_input_sequence'].append('S')
        
    if mus2.button('रे॒'):
        st.session_state['user_input_sequence'].append('r')
        
    if mus3.button('रे '):
        st.session_state['user_input_sequence'].append('R')
        
    if mus4.button('ग॒ '):
        st.session_state['user_input_sequence'].append('g')
        
    if mus5.button('ग '):
        st.session_state['user_input_sequence'].append('G')
        
    if mus6.button('म'):
        st.session_state['user_input_sequence'].append('m')
        
    if mus7.button('म॑'):
        st.session_state['user_input_sequence'].append('M')
        
    if mus8.button('प'):
        st.session_state['user_input_sequence'].append('P')
        
    if mus9.button('ध॒'):
        st.session_state['user_input_sequence'].append('d')
        
    if mus10.button('ध'):
        st.session_state['user_input_sequence'].append('D')
        
    if mus11.button('नि॒'):
        st.session_state['user_input_sequence'].append('n')
        
    if mus12.button('नि'):
        st.session_state['user_input_sequence'].append('N')
        
    if mus13.button('सं'):
        st.session_state['user_input_sequence'].append('S^')
    
    if mus14.button(' '):
        st.session_state['user_input_sequence'].append('''''')
        
    if mus15.button('Del'):
        st.session_state['user_input_sequence'].pop()
        
    if mus16.button('Clear all'):
        st.session_state['user_input_sequence'].clear()
        
    
    music1, music2= st.columns(2)
    music1.text_input('Note sequence', st.session_state['user_input_sequence'])
    
    
    
    if len(st.session_state['user_input_sequence']) >0 :
        if st.sidebar.checkbox("Real time generate music"):
            
            note_list=[]
            
            for i_note in st.session_state['user_input_sequence']:
                note_list.append(notes_freq_dict[i_note])
                
                
            notelist_final=[]
            note_name_list_final=[]
            for i_note in note_list:
                for i in range(beats_per_note):
                    notelist_final.append(i_note)
            
                
                    
            df_music_generate = pd.Series(notelist_final, index =[i_ for i_ in range(len(notelist_final))])
            #df_music_generate = df_music_generate.interpolate(method='cubic')
            #df_music_generate= df_music_generate.rolling(beats_per_note).mean()
            
            
            if poly_n != 0:
                model = np.poly1d(np.polyfit(df_music_generate.index.tolist(), df_music_generate.tolist(), poly_n))
                polyline=df_music_generate.index.values.tolist()
                y_val= model(polyline)
            else:
                y_val= df_music_generate.tolist()
            

            x_val= df_music_generate.index.tolist()
            
            
            
            
            ######################################## plot graph ################################
            fig= px.line(x= x_val, y= y_val, width= 500, height=400, template="plotly_white")
            #fig= px.line(x= x_val, y= y_val,template="plotly_white")
            fig.update_layout(xaxis_title="Time", yaxis_title= "frequency" )
            fig.update_traces(line=dict( color="Red", width=3.5))
            music1.plotly_chart(fig)
            
            
            ######################################## Code to edit the sequence ###########################
            df_edit_seq= pd.DataFrame( columns= ['Time', 'Frequency'])
            df_edit_seq['Time']= x_val
            df_edit_seq['Frequency']= y_val
            i_splice= music2.slider(
                 'Selection',
                 min(x_val), max(x_val), (0, 1), step=1)
            
            df_edit_seq_small= df_edit_seq[(df_edit_seq['Time'] >= i_splice[0]) & (df_edit_seq['Time'] <=i_splice[1] )]
            
            
            fig= px.line( x= df_edit_seq['Time'], y= df_edit_seq['Frequency'], width= 500, height=400, template="plotly_white")
            fig.update_layout(xaxis_title="Time", yaxis_title= "frequency" ,
                              shapes=[
                                dict(
                                    type="rect",
                                    xref="x",
                                    yref="y",
                                    x0= i_splice[0],
                                    y0=min(y_val),
                                    x1=i_splice[1],
                                    y1=max(y_val),
                                    fillcolor="lightgray",
                                    opacity=0.4,
                                    line_width=0,
                                    layer="below"
                                )]
                              
                              )
            fig.update_traces(line=dict( color="BLue", width=3.5))
            music2.plotly_chart(fig)
            
            fig= px.line( x= df_edit_seq_small['Time'], y= df_edit_seq_small['Frequency'], width= 500, height=400, template="plotly_white")
            fig.update_layout(xaxis_title="Time", yaxis_title= "frequency" )
            fig.update_traces(line=dict( color="BLue", width=3.5))
            music2.plotly_chart(fig)
                        
            
            
            ############################################ Code to play the music ################################
            amplitude = 4096*3 #arbitrary value
            duration=0.1 #this is from the data: sample_data['Time'][5]-sample_data['Time'][4]
            samplerate = 44100
            original_freq= y_val #notelist_final
            # fit_freq= df_result['Amplitude_fit'].values.tolist()
            
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
            write('generate_music.wav', samplerate, data.astype(np.int16))
            
            audio_file = open('generate_music.wav', 'rb')
            audio_bytes = audio_file.read()
            music1.audio(audio_bytes, format='audio/wav')
    
    
######################################################### Bezier Curve ##################################

elif i_page=='Bezier Curve':
    st.write("Bezier")
    t_points = np.arange(0, 1, 0.01) #................................. Creates an iterable list from 0 to 1.
    points1 = np.array([[0, 0], [0, 5],[1,2],[1,5], [2, 2]]) #.... Creates an array of coordinates.
    curve1 = Bezier.Curve(t_points, points1) #......................... Returns an array of coordinates.
    
    fig = plt.figure(figsize = (9, 7))
    plt.plot(
     	curve1[:, 0],   # x-coordinates.
     	curve1[:, 1]    # y-coordinates.
    )
    plt.plot(
     	points1[:, 0],  # x-coordinates.
     	points1[:, 1],  # y-coordinates.
     	'ro:'           # Styling (red, circles, dotted).
    )
    plt.grid()
    st.pyplot(fig) 
    
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider, Button
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import scipy.interpolate as inter
    import numpy as np
    
    
    
    func = lambda x: 0.1*x**2
    
    #get a list of points to fit a spline to as well
    N = 10
    xmin = 0 
    xmax = 10 
    x = np.linspace(xmin,xmax,N)
    
    #spline fit
    yvals = func(x)
    spline = inter.InterpolatedUnivariateSpline (x, yvals)
    
    #figure.subplot.right
    mpl.rcParams['figure.subplot.right'] = 0.8
    
    #set up a plot
    fig,axes = plt.subplots(1,1,figsize=(9.0,8.0),sharex=True)
    ax1 = axes
    
    
    pind = None #active point
    epsilon = 5 #max pixel distance
    
    def update(val):
        global yvals
        global spline
        # update curve
        for i in np.arange(N):
          yvals[i] = sliders[i].val 
        l.set_ydata(yvals)
        spline = inter.InterpolatedUnivariateSpline (x, yvals)
        m.set_ydata(spline(X))
        # redraw canvas while idle
        fig.canvas.draw_idle()
    
    def reset(event):
        global yvals
        global spline
        #reset the values
        yvals = func(x)
        for i in np.arange(N):
          sliders[i].reset()
        spline = inter.InterpolatedUnivariateSpline (x, yvals)
        l.set_ydata(yvals)
        m.set_ydata(spline(X))
        # redraw canvas while idle
        fig.canvas.draw_idle()
    
    def button_press_callback(event):
        'whenever a mouse button is pressed'
        global pind
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        #print(pind)
        pind = get_ind_under_point(event)    
    
    def button_release_callback(event):
        'whenever a mouse button is released'
        global pind
        if event.button != 1:
            return
        pind = None
    
    def get_ind_under_point(event):
        'get the index of the vertex under point if within epsilon tolerance'
    
        # display coords
        #print('display x is: {0}; display y is: {1}'.format(event.x,event.y))
        t = ax1.transData.inverted()
        tinv = ax1.transData 
        xy = t.transform([event.x,event.y])
        #print('data x is: {0}; data y is: {1}'.format(xy[0],xy[1]))
        xr = np.reshape(x,(np.shape(x)[0],1))
        yr = np.reshape(yvals,(np.shape(yvals)[0],1))
        xy_vals = np.append(xr,yr,1)
        xyt = tinv.transform(xy_vals)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]
    
        #print(d[ind])
        if d[ind] >= epsilon:
            ind = None
        
        #print(ind)
        return ind
    
    def motion_notify_callback(event):
        'on mouse movement'
        global yvals
        if pind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        
        #update yvals
        #print('motion x: {0}; y: {1}'.format(event.xdata,event.ydata))
        yvals[pind] = event.ydata 
    
        # update curve via sliders and draw
        sliders[pind].set_val(yvals[pind])
        fig.canvas.draw_idle()
    
    X = np.arange(0,xmax+1,0.1)
    ax1.plot (X, func(X), 'k--', label='original')
    l, = ax1.plot (x,yvals,color='k',linestyle='none',marker='o',markersize=8)
    m, = ax1.plot (X, spline(X), 'r-', label='spline')
    
    
    
    ax1.set_yscale('linear')
    ax1.set_xlim(0, xmax)
    ax1.set_ylim(0,xmax)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    ax1.yaxis.grid(True,which='minor',linestyle='--')
    ax1.legend(loc=2,prop={'size':22})
    
    sliders = []
    
    for i in np.arange(N):
    
        axamp = plt.axes([0.84, 0.8-(i*0.05), 0.12, 0.02])
        # Slider
        s = Slider(axamp, 'p{0}'.format(i), 0, 10, valinit=yvals[i])
        sliders.append(s)
    
        
    for i in np.arange(N):
        #samp.on_changed(update_slider)
        sliders[i].on_changed(update)
    
    axres = plt.axes([0.84, 0.8-((N)*0.05), 0.12, 0.02])
    bres = Button(axres, 'Reset')
    bres.on_clicked(reset)
    
    fig.canvas.mpl_connect('button_press_event', button_press_callback)
    fig.canvas.mpl_connect('button_release_event', button_release_callback)
    fig.canvas.mpl_connect('motion_notify_event', motion_notify_callback)
    
    st.pyplot(fig) 
    
    
    
    
    