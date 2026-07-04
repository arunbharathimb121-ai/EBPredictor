import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from bill_utils import (
    simulate_dataset,
    compute_bill_generic_india,
    compute_bill_tn,
    estimate_units_from_devices,
    classify_dependency,
    parse_devices_textarea
)
from ml_utils import (
    load_or_train_model,
    get_temperature_for_city
)

st.set_page_config(page_title='Electricity Bill Chatbot', layout='centered')
st.title('🔌 Electricity Bill Predictor — Chatbot (Enhanced)')
st.caption('Auto weather for Indian cities + fans per room + tariff-accurate bills (Generic India or Tamil Nadu).')

now = datetime.now()
st.sidebar.write('Current date & time:')
st.sidebar.write(now.strftime('%Y-%m-%d %H:%M:%S'))

with st.spinner('Preparing model (demo)...'):
    df = simulate_dataset(n_months=24)
    model, scaler, feature_names = load_or_train_model(df)

st.markdown('---')
st.header('Tell me about your home')

with st.form('home_form'):
    col1, col2 = st.columns(2)
    with col1:
        people = st.number_input('Number of people', 1, 20, 3)
        rooms = st.number_input('Number of rooms', 1, 20, 3)
        city_type = st.selectbox('City type', ['Urban', 'Rural'])
        city_name = st.text_input('City name (for weather lookup)', 'Chennai')
    with col2:
        efficiency = st.slider('Appliance efficiency (0.6–1.0)', 0.6, 1.0, 0.85)
        tariff_choice = st.selectbox('Tariff', ['Generic India (sample)', 'Tamil Nadu (TANGEDCO Domestic)'])
        tn_cycle = st.selectbox('TN Billing Cycle', ['Monthly', 'Bi-monthly'])

    st.markdown('**Devices** — format: `Name, hours_per_day, always_on(yes/no)`')
    st.caption('Example: AC,8,yes  OR  Fridge,24,yes  OR  TV,3,no')
    devices_text = st.text_area('Devices', value='Fridge,24,yes\nAC,8,yes\nTV,4,no')
    submitted = st.form_submit_button('Predict Bill')

if not submitted:
    st.stop()

# ---------- Prediction workflow ----------
temp_c, place_resolved, geo = get_temperature_for_city(city_name)
if temp_c is None:
    st.warning(f'Could not auto-detect temperature for {city_name}. Enter manually.')
    avg_temp = st.number_input('Average temperature (°C)', 26.0, step=0.5)
else:
    st.info(f'Fetched temperature for {place_resolved}: {temp_c:.1f} °C')
    avg_temp = temp_c

devices = parse_devices_textarea(devices_text)
if not devices:
    st.error('Please list at least one device.')
    st.stop()

# Add one fan per room
for i in range(int(rooms)):
    devices.append({'name': f'Fan_{i+1}', 'hours': 8, 'always_on': False})

for d in devices:
    d['dependency'] = classify_dependency(d)

dep_df = pd.DataFrame([{
    'Device': d['name'], 'Hours/day': d['hours'],
    'Always on': d['always_on'], 'Dependency': d['dependency']
} for d in devices])
st.subheader('Device dependency')
st.table(dep_df)

fully = sum(d['dependency']=='fully-dependent' for d in devices)
partial = sum(d['dependency']=='partially-dependent' for d in devices)
non = sum(d['dependency']=='non-dependent' for d in devices)

est_units = estimate_units_from_devices(devices)
st.info(f'Estimated monthly units (from devices + fans): ~{est_units} kWh')

month = now.month
ac_flag = any('ac' in d['name'].lower() for d in devices)
city_flag = 1 if city_type == 'Urban' else 0
X = np.array([[month, avg_temp, est_units, people, int(ac_flag),
               city_flag, efficiency, fully, partial, non]])
Xs = scaler.transform(X)
ml_pred_value = float(model.predict(Xs).flatten()[0])

st.markdown('---')
st.subheader('Tariff-based Bill')

if tariff_choice.startswith('Tamil'):
    tn = compute_bill_tn(est_units, cycle=tn_cycle)
    st.metric(f'TN {tn["cycle_name"]} Bill (₹)', f"{tn['total_cycle']:,}")
    st.caption(f"Monthly ≈ ₹{tn['equivalent_monthly']:,} | Units {int(tn['units_cycle'])} | Fixed ₹{tn['fixed']}")
    st.table(pd.DataFrame(tn['breakdown']))
else:
    total, br, fixed = compute_bill_generic_india(est_units, city_type)
    st.metric('Generic India Bill (₹)', f"{total:,}")
    st.caption(f"Fixed ₹{fixed} | Slabs 0–100@3  101–300@4.5  300+@6.5")
    st.table(pd.DataFrame(br))

with st.expander('ML Model Estimate (demo)'):
    st.metric('Predicted Bill (₹)', f"{ml_pred_value:.0f}")


