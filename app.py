import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- Path Resolution Helper ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(APP_DIR)

def resolve_file(filename):
    for candidate in [os.path.join(APP_DIR, filename), os.path.join(PARENT_DIR, filename)]:
        if os.path.exists(candidate):
            return candidate
    return filename

# --- Page Config ---
st.set_page_config(
    page_title="Smart Parking Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px 22px;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.06);
        min-height: 125px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #2c3e50;
        line-height: 1.1;
        white-space: nowrap;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data & Model ---
@st.cache_data
def load_data():
    csv_file = resolve_file('smart_parking_usage_occupancy_analytics.csv')
    df = pd.read_csv(csv_file)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df_valid = df[df['occupied_slots'] <= df['total_slots']].copy()
    df_valid['hour'] = df_valid['date_time'].dt.hour
    df_valid['day_of_week'] = df_valid['date_time'].dt.dayofweek
    df_valid['is_weekend'] = df_valid['day_of_week'].isin([5, 6]).astype(int)
    return df_valid

@st.cache_resource
def load_model():
    model_file = resolve_file('best_parking_model_pipeline.joblib')
    return joblib.load(model_file)

try:
    df = load_data()
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model or data. Please ensure 'smart_parking_usage_occupancy_analytics.csv' and 'best_parking_model_pipeline.joblib' are in the directory. Error: {e}")
    model_loaded = False

# --- Sidebar Navigation ---
st.sidebar.title("🚗 Smart Parking ")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Overview Dashboard", " Prediction Engine"])

if model_loaded:
    if page == "Overview Dashboard":
        st.title("📊 Parking Overview Dashboard")
        st.markdown("Real-time and historical analytics of parking zones.")
        
        # --- Top KPIs ---
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        total_capacity = df['total_slots'].sum()
        current_occupied = df['occupied_slots'].sum()
        avg_occupancy = df['occupancy_rate_percent'].mean()
        
        # Calculate Peak Time
        peak_hour = df.groupby('hour')['occupied_slots'].mean().idxmax()
        peak_time_str = f"{peak_hour}:00 - {peak_hour+1}:00"
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Valid Capacity</div>
                <div class="metric-value">{total_capacity:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #28a745;">
                <div class="metric-label">Total Occupied</div>
                <div class="metric-value">{current_occupied:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #ffc107;">
                <div class="metric-label">Avg Occupancy Rate</div>
                <div class="metric-value">{avg_occupancy:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #dc3545;">
                <div class="metric-label">Global Peak Time</div>
                <div class="metric-value">{peak_time_str}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- Charts ---
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Zone-Wise Occupancy Breakdown")
            zone_df = df.groupby('zone')[['total_slots', 'occupied_slots']].sum().reset_index()
            
            fig = px.pie(zone_df, values='occupied_slots', names='zone', hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Teal)
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with chart_col2:
            st.subheader("Hourly Parking Trends")
            hourly_df = df.groupby('hour')['occupied_slots'].mean().reset_index()
            fig2 = px.area(hourly_df, x='hour', y='occupied_slots', 
                           labels={'hour': 'Time of Day (24h)', 'occupied_slots': 'Avg Vehicles'},
                           color_discrete_sequence=['#3498db'])
            fig2.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig2, use_container_width=True)
            
        # Vehicle Type Distribution
        st.subheader("Vehicle Type Distribution")
        veh_df = df['vehicle_type'].value_counts().reset_index()
        veh_df.columns = ['Vehicle Type', 'Count']
        fig3 = px.bar(veh_df, x='Vehicle Type', y='Count', 
                      color='Vehicle Type', text_auto=True,
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig3, use_container_width=True)


    elif page == " Prediction Engine":
        st.title("🤖 Parking Occupancy Predictor")
        st.markdown("Enter operational details to predict the number of occupied parking slots using the Random Forest Regressor pipeline.")
        
        with st.form("prediction_form"):
            st.subheader("Input Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                input_zone = st.selectbox("Parking Zone", ['Mall', 'Hospital', 'Office', 'Residential', 'Commercial'])
                input_veh = st.selectbox("Primary Vehicle Type", ['Car', 'Bike', 'Electric', 'SUV'])
                # Logical constraints applied
                input_slots = st.number_input("Total Zone Capacity (Slots)", min_value=10, max_value=2000, value=150, step=10)
                
            with col2:
                # Hour constrained 0-23
                input_hour = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=datetime.now().hour, step=1)
                # Day constrained 0-6
                input_day = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=datetime.now().weekday(), step=1)
                input_rate = st.slider("Current/Expected Occupancy Rate (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
            
            is_weekend = 1 if input_day in [5, 6] else 0
            
            submit_button = st.form_submit_button(label="Predict Occupancy")
            
        if submit_button:
            # Prepare input dataframe aligning with pipeline expectations
            input_data = pd.DataFrame([{
                'zone': input_zone,
                'vehicle_type': input_veh,
                'total_slots': input_slots,
                'hour': input_hour,
                'day_of_week': input_day,
                'is_weekend': is_weekend,
                'occupancy_rate_percent': input_rate
            }])
            
            try:
                prediction = model.predict(input_data)[0]
                
                # Logic check: Cannot exceed total slots
                prediction = min(prediction, input_slots)
                # Logic check: Cannot be negative
                prediction = max(0, prediction)
                
                st.markdown("---")
                st.subheader("Prediction Result")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric(label="Predicted Occupied Slots", value=int(round(prediction)))
                    
                with res_col2:
                    empty = int(input_slots) - int(round(prediction))
                    st.metric(label="Predicted Available Slots", value=empty, delta=empty, delta_color="normal")
                    
                st.success("Prediction generated successfully utilizing the Pipeline.")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
