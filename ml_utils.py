import os
import pickle
import joblib
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import streamlit as st

def build_and_train_model(X,y,model_path='model.h5',scaler_path='scaler.pkl'):
    st.info('Training TensorFlow model...')
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    model=Sequential([Dense(32,activation='relu',input_shape=(Xs.shape[1],)),
                      Dense(16,activation='relu'),Dense(1)])
    model.compile(optimizer=Adam(0.01),loss='mse')
    model.fit(Xs,y,epochs=40,batch_size=32,verbose=0)
    model.save(model_path); joblib.dump(sc,scaler_path)
    st.success('Model trained.')
    return model,sc

def load_or_train_model(df):
    model_path,scaler_path='model.h5','scaler.pkl'
    feats=['month','avg_temp','units','people','ac','city','efficiency',
           'fully_dep_count','partial_dep_count','non_dep_count']
    X,y=df[feats],df['bill']
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            return load_model(model_path), joblib.load(scaler_path), feats
        except Exception:
            st.warning('Reload failed – retraining.')
    return build_and_train_model(X,y,model_path,scaler_path)+(feats,)

# ---------- Weather ----------
@st.cache_data(ttl=900)
def get_temperature_for_city(city):
    try:
        g=requests.get("https://geocoding-api.open-meteo.com/v1/search",
                       params={'name':city,'count':5,'language':'en','format':'json'},timeout=8).json()
        results=g.get('results',[])
        target=next((r for r in results if r.get('country_code')=='IN'), results[0])
        lat,lon=target['latitude'],target['longitude']
        w=requests.get("https://api.open-meteo.com/v1/forecast",
                       params={'latitude':lat,'longitude':lon,'current_weather':True,'timezone':'auto'},timeout=8).json()
        return float(w['current_weather']['temperature']), f"{target['name']}, {target.get('admin1','')}", target
    except Exception:
        return None,None,None
