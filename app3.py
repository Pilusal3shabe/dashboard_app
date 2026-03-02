import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
import plotly.express as px
from scipy.optimize import minimize
from scipy import stats
import warnings
import base64
from pathlib import Path
warnings.filterwarnings('ignore')


# ============================================================================
# TLOWANA RESOURCES BRANDING - COLOR THEME
# ============================================================================
TLOWANA_GREEN = '#6B7C2E'      # Primary olive/army green from logo
TLOWANA_DARK = '#2C2C2C'       # Black/dark gray from logo
TLOWANA_LIGHT_GREEN = '#8B9D46' # Light green accent
TLOWANA_GRAY = '#7F7F7F'       # Medium gray

CHART_COLORS = {
    'primary': TLOWANA_GREEN,
    'secondary': '#D4AF37',     # Gold accent
    'success': '#4CAF50',
    'warning': '#FF9800',
    'danger': '#F44336',
    'info': '#2196F3'
}

# 1. Page Configuration
st.set_page_config(page_title="Tlowana Resources - Combustion Dashboard", layout="wide")


# ============================================================================
# TLOWANA RESOURCES BRANDING - CUSTOM CSS
# ============================================================================
st.markdown(f"""
    <style>
    /* Tlowana brand colors */
    :root {{
        --tlowana-green: {TLOWANA_GREEN};
        --tlowana-dark: {TLOWANA_DARK};
    }}
    
    /* Sidebar styling with gradient */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {TLOWANA_GREEN} 0%, {TLOWANA_DARK} 100%);
    }}
    
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    
    /* Logo container */
    .logo-container {{
        text-align: center;
        padding: 10px;
        background: white;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Main header styling */
    .main-header {{
        background: linear-gradient(135deg, {TLOWANA_GREEN} 0%, {TLOWANA_DARK} 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .main-header h1 {{
        color: white !important;
        margin: 0;
        font-size: 2.5em;
        font-weight: bold;
    }}
    
    .main-header p {{
        color: #E8E8E8 !important;
        margin: 5px 0 0 0;
        font-size: 1.1em;
    }}
    
    /* Metric values in Tlowana green */
    [data-testid="stMetricValue"] {{
        color: {TLOWANA_GREEN} !important;
        font-weight: bold;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {TLOWANA_GREEN};
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }}
    
    .stButton>button:hover {{
        background-color: {TLOWANA_LIGHT_GREEN};
    }}
    
    /* Dividers */
    hr {{
        border-color: {TLOWANA_GREEN};
        opacity: 0.3;
    }}
    </style>
""", unsafe_allow_html=True)



# ============================================================================
# TLOWANA RESOURCES BRANDING - LOGO DISPLAY
# ============================================================================
def display_logo():
    """Display Tlowana Resources logo in sidebar"""
    try:
        logo_path = Path("Company_Logo.jpg")
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                logo_data = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<div class="logo-container"><img src="data:image/jpeg;base64,{logo_data}" width="250"></div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                f'<div class="logo-container"><h2 style="color: {TLOWANA_GREEN}; margin: 0;">TLOWANA RESOURCES</h2></div>',
                unsafe_allow_html=True
            )
    except Exception:
        st.markdown(
            f'<div class="logo-container"><h2 style="color: {TLOWANA_GREEN}; margin: 0;">TLOWANA RESOURCES</h2></div>',
            unsafe_allow_html=True
        )

# 2. Optimized Data Loading with Caching
@st.cache_data(ttl=600)
def load_data():
    df = pd.read_excel('Data_25.xlsx')
    df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ============================================================================
# STATISTICAL PROCESS CONTROL FUNCTIONS
# ============================================================================

def calculate_cp_cpk(data, lsl, usl, target=None):
    """Calculate Process Capability indices Cp and Cpk"""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if std == 0:
        return None, None, None, None
    
    # Cp - Process Capability (doesn't consider centering)
    cp = (usl - lsl) / (6 * std)
    
    # Cpk - Process Capability Index (considers centering)
    cpu = (usl - mean) / (3 * std)
    cpl = (mean - lsl) / (3 * std)
    cpk = min(cpu, cpl)
    
    # Pp and Ppk (for overall performance)
    pp = (usl - lsl) / (6 * std)
    ppk = cpk  # Simplified
    
    return cp, cpk, cpu, cpl

def calculate_control_limits_xbar_r(data, subgroup_size=5):
    """Calculate X-bar and R chart control limits"""
    # Constants for control charts
    A2_constants = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
    D3_constants = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
    D4_constants = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
    
    if subgroup_size not in A2_constants:
        subgroup_size = 5
    
    A2 = A2_constants[subgroup_size]
    D3 = D3_constants[subgroup_size]
    D4 = D4_constants[subgroup_size]
    
    # Create subgroups
    n_subgroups = len(data) // subgroup_size
    subgroups = [data[i*subgroup_size:(i+1)*subgroup_size] for i in range(n_subgroups)]
    
    # Calculate means and ranges for each subgroup
    xbar_values = [np.mean(subgroup) for subgroup in subgroups]
    r_values = [np.max(subgroup) - np.min(subgroup) for subgroup in subgroups]
    
    # Calculate centerlines
    xbar_centerline = np.mean(xbar_values)
    r_centerline = np.mean(r_values)
    
    # Calculate control limits
    xbar_ucl = xbar_centerline + A2 * r_centerline
    xbar_lcl = xbar_centerline - A2 * r_centerline
    
    r_ucl = D4 * r_centerline
    r_lcl = D3 * r_centerline
    
    return {
        'xbar_values': xbar_values,
        'r_values': r_values,
        'xbar_centerline': xbar_centerline,
        'xbar_ucl': xbar_ucl,
        'xbar_lcl': xbar_lcl,
        'r_centerline': r_centerline,
        'r_ucl': r_ucl,
        'r_lcl': r_lcl
    }

def calculate_control_limits_imr(data):
    """Calculate I-MR (Individual-Moving Range) chart control limits"""
    # Calculate moving ranges
    moving_ranges = [abs(data[i] - data[i-1]) for i in range(1, len(data))]
    
    # Calculate mean of individuals and moving ranges
    x_bar = np.mean(data)
    mr_bar = np.mean(moving_ranges)
    
    # Constants for I-MR charts
    d2 = 1.128  # for n=2 (moving range of 2)
    D3 = 0      # for n=2
    D4 = 3.267  # for n=2
    
    # Control limits for Individuals chart
    i_ucl = x_bar + 2.66 * mr_bar
    i_lcl = x_bar - 2.66 * mr_bar
    
    # Control limits for Moving Range chart
    mr_ucl = D4 * mr_bar
    mr_lcl = D3 * mr_bar
    
    return {
        'individuals': data,
        'moving_ranges': moving_ranges,
        'x_bar': x_bar,
        'i_ucl': i_ucl,
        'i_lcl': i_lcl,
        'mr_bar': mr_bar,
        'mr_ucl': mr_ucl,
        'mr_lcl': mr_lcl
    }

# ============================================================================
# ML MODEL TRAINING FUNCTIONS
# ============================================================================

@st.cache_resource
def train_loi_predictor(df):
    """Train Random Forest model to predict LOI"""
    features = [
        '3 Pt Average Sinterting Temp', 
        '3PT Average Burner Temp',
        '3 Pt Average Inlet Temp',
        '3 Pt Average Gas setting (% Openning)',
        '3 Pt Average Gas Consumption (m^3)',
        'VSD speed',
        'Air Flow (%)'
    ]
    
    target = '3 Pt Average LOI (%)'
    
    df_clean = df[features + [target]].dropna()
    
    if len(df_clean) < 10:
        return None, None, None
    
    X = df_clean[features]
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    return model, metrics, features

@st.cache_resource
def train_gas_predictor(df):
    """Train model to predict gas consumption"""
    features = [
        '3 Pt Average Sinterting Temp',
        '3PT Average Burner Temp',
        '3 Pt Average Inlet Temp',
        '3 Pt Average Gas setting (% Openning)',
        'VSD speed',
        'Air Flow (%)'
    ]
    
    target = '3 Pt Average Gas Consumption (m^3)'
    
    df_clean = df[features + [target]].dropna()
    
    if len(df_clean) < 10:
        return None, None, None
    
    X = df_clean[features]
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    return model, metrics, features

@st.cache_resource
def train_quality_classifier(df):
    """Train classifier to predict if quality targets will be met"""
    features = [
        '3 Pt Average Sinterting Temp',
        '3PT Average Burner Temp',
        '3 Pt Average Inlet Temp',
        '3 Pt Average Gas setting (% Openning)',
        'VSD speed',
        'Air Flow (%)'
    ]
    
    df_clean = df[features + ['3 Pt Average LOI (%)', '3 Pt Average Reactivity (sec)']].dropna()
    
    if len(df_clean) < 10:
        return None, None, None
    
    df_clean['Quality_Good'] = ((df_clean['3 Pt Average LOI (%)'] < 5) & 
                                  (df_clean['3 Pt Average Reactivity (sec)'] < 90)).astype(int)
    
    X = df_clean[features]
    y = df_clean['Quality_Good']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = (y_pred == y_test).mean()
    
    metrics = {
        'Accuracy': accuracy,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return model, metrics, features

@st.cache_resource
def train_anomaly_detector(df):
    """Train Isolation Forest for anomaly detection"""
    features = [
        '3 Pt Average Sinterting Temp',
        '3PT Average Burner Temp',
        '3 Pt Average Inlet Temp',
        '3 Pt Average LOI (%)',
        '3 Pt Average Gas Consumption (m^3)',
        'VSD speed',
        'Air Flow (%)'
    ]
    
    df_clean = df[features].dropna()
    
    if len(df_clean) < 10:
        return None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_scaled)
    
    return model, scaler

# 3. Sidebar Navigation & Filters
with st.sidebar:
    display_logo()
    st.markdown("---")

st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select View",
    ["📊 Main Dashboard", "🤖 ML Analytics", "📈 Process Capability (Cp/Cpk)", "📉 Control Charts", "🔬 ML Model Details"]
)

st.sidebar.divider()
st.sidebar.header("Filters & Slicers")

min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()
date_range = st.sidebar.date_input("📅 Select Date Range", [min_date, max_date])

if 'Process stage' in df.columns:
    process_stages = ['All'] + sorted(list(df['Process stage'].dropna().unique()))
    selected_stage = st.sidebar.selectbox("🔧 Process Stage", process_stages)
else:
    selected_stage = 'All'

st.sidebar.divider()

# Filter data
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0] if isinstance(date_range, (list, tuple)) else date_range

mask = (df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)

if selected_stage != 'All' and 'Process stage' in df.columns:
    mask = mask & (df['Process stage'] == selected_stage)

filtered_df = df.loc[mask]

if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters.")
    st.stop()

# Train all models
with st.spinner("🤖 Training ML models..."):
    loi_model, loi_metrics, loi_features = train_loi_predictor(df)
    gas_model, gas_metrics, gas_features = train_gas_predictor(df)
    quality_model, quality_metrics, quality_features = train_quality_classifier(df)
    anomaly_model, anomaly_scaler = train_anomaly_detector(df)

# ============================================================================
# PAGE 1: MAIN DASHBOARD
# ============================================================================

if page == "📊 Main Dashboard":
    st.markdown(f"""
        <div class="main-header">
            <h1>🔥 Combustion Area Analysis Dashboard</h1>
            <p>Analysis Period: {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}</p>
        </div>
    """, unsafe_allow_html=True)
    
    if selected_stage != 'All':
        st.caption(f"🏷️ Active Filter: Process Stage = {selected_stage}")
    
    st.divider()
    
    # KPIs
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container(border=True):
            avg_sintering = filtered_df['3 Pt Average Sinterting Temp'].mean()
            target_temp = filtered_df['Desired Temp'].mean()
            st.metric(
                "Avg Sintering Temp", 
                f"{avg_sintering:.1f} °C", 
                delta=f"{avg_sintering - target_temp:.1f}° vs Target"
            )
    
    with col2:
        with st.container(border=True):
            avg_loi = filtered_df['3 Pt Average LOI (%)'].mean()
            st.metric(
                "Average LOI", 
                f"{avg_loi:.2f}%", 
                delta="Good" if avg_loi < 5 else "High", 
                delta_color="normal" if avg_loi < 5 else "inverse"
            )
    
    with col3:
        with st.container(border=True):
            avg_gas = filtered_df['3 Pt Average Gas Consumption (m^3)'].mean()
            st.metric("Avg Gas Consumption", f"{avg_gas:.2f} m³")
    
    with col4:
        with st.container(border=True):
            avg_reactivity = filtered_df['3 Pt Average Reactivity (sec)'].mean()
            st.metric(
                "Avg Reactivity", 
                f"{avg_reactivity:.1f} sec",
                delta="Good" if avg_reactivity < 90 else "High",
                delta_color="normal" if avg_reactivity < 90 else "inverse"
            )
    
    st.divider()
    
    # CHART 1: VSD Setpoint VS Kiln Temperature Profile
    st.header("VSD Setpoint VS Kiln Temperature Profile")
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['3 Pt Average Sinterting Temp'],
                              name='3 Pt Average Sinterting Temp', line=dict(color='red', width=2), yaxis='y1'))
    fig1.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['Desired Temp'],
                              name='Desired Temp', line=dict(color='green', width=2), yaxis='y1'))
    fig1.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['3PT Average Burner Temp'],
                              name='3PT Average Burner Temp', line=dict(color='red', width=2, dash='dash'), yaxis='y1'))
    fig1.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['3 Pt Average Inlet Temp'],
                              name='3 Pt Average Inlet Temp', line=dict(color='red', width=1, dash='dot'), yaxis='y1'))
    fig1.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['Burner Temp (Target)'],
                              name='Burner Temp (Target)', line=dict(color='green', width=2), yaxis='y1'))
    fig1.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['Inlet Temp (Target)'],
                              name='Inlet Temp (Target)', line=dict(color='green', width=2), yaxis='y1'))
    fig1.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['VSD speed'],
                              name='VSD speed', line=dict(color='blue', width=2), yaxis='y2'))
    
    fig1.update_layout(
        xaxis=dict(title='DATE/TIME'),
        yaxis=dict(title=dict(text='Temperature (Degree Celsius) / Target Temps (Degree Celsius)', font=dict(color='green')),
                   side='left', range=[0, 1200]),
        yaxis2=dict(title=dict(text='VSD kiln speed setpoint (%)', font=dict(color='blue')),
                    overlaying='y', side='right', range=[0, 70]),
        height=500, template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5)
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    st.divider()
    
    # CHART 2: Gas Consumption vs Kiln Temperature Profile
    st.header("Gas Consumption vs Kiln Temperature Profile")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['3 Pt Average Sinterting Temp'],
                              name='3 Pt Average Sinterting Temp', line=dict(color='red', width=2), yaxis='y1'))
    fig2.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['Desired Temp'],
                              name='Desired Temp', line=dict(color='green', width=2), yaxis='y1'))
    fig2.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['3PT Average Burner Temp'],
                              name='3PT Average Burner Temp', line=dict(color='red', width=2, dash='dash'), yaxis='y1'))
    fig2.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['Burner Temp (Target)'],
                              name='Burner Temp (Target)', line=dict(color='green', width=2), yaxis='y1'))
    fig2.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['3 Pt Average Inlet Temp'],
                              name='3 Pt Average Inlet Temp', line=dict(color='red', width=1, dash='dot'), yaxis='y1'))
    fig2.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['Inlet Temp (Target)'],
                              name='Inlet Temp (Target)', line=dict(color='green', width=2), yaxis='y1'))
    fig2.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['3 Pt Average Gas Consumption (m^3)'],
                              name='3 Pt Average Gas Consumption (m^3)', line=dict(color='orange', width=3), yaxis='y2'))
    
    fig2.update_layout(
        xaxis=dict(title='DATE/TIME'),
        yaxis=dict(title=dict(text='Kiln Temperature (Degree Celsius) / Target Temp (Degree Celsius)', font=dict(color='green')),
                   side='left', range=[0, 1200]),
        yaxis2=dict(title=dict(text='Gas Consumption (m^3/hour)', font=dict(color='orange')),
                    overlaying='y', side='right', range=[0, 1400]),
        height=500, template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    st.divider()
    
    # CHART 3: Impact of Kiln Speed on LOI
    st.header("Impact of Kiln Speed on LOI")
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=[5] * len(filtered_df),
                              fill=None, mode='lines', line=dict(width=0), showlegend=False, yaxis='y1'))
    fig3.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=[0] * len(filtered_df),
                              fill='tonexty', mode='lines', line=dict(width=0),
                              fillcolor='rgba(144, 238, 144, 0.5)', name='Target LOI < 5', yaxis='y1'))
    fig3.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['3 Pt Average LOI (%)'],
                              name='Average LOI (%) / Target LOI', line=dict(color='blue', width=2), yaxis='y1'))
    fig3.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['VSD speed'],
                              name='VSD Speed Setpoint (%)', line=dict(color='orange', width=2), yaxis='y2'))
    
    fig3.update_layout(
        xaxis=dict(title='DATE/TIME'),
        yaxis=dict(title=dict(text='Average LOI (%) / Target LOI (%)', font=dict(color='blue')),
                   side='left', range=[0, 30]),
        yaxis2=dict(title=dict(text='VSD Speed Setpoint (%)', font=dict(color='orange')),
                    overlaying='y', side='right', range=[0, 14]),
        height=500, template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5)
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    st.divider()
    
    # CHARTS 4 & 5
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Air Flow vs Gas Consumption")
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['3 Pt Average Gas Consumption (m^3)'],
                                  name='3 Pt Average Gas Consumption (m^3)', line=dict(color='orange', width=2), yaxis='y1'))
        fig4.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['Air Flow (%)'],
                                  name='Air Flow (%)', line=dict(color='blue', width=2), yaxis='y2'))
        
        fig4.update_layout(
            xaxis=dict(title='DATE/TIME'),
            yaxis=dict(title=dict(text='Gas consumption (m^3/hour)', font=dict(color='orange')),
                       side='left', range=[0, 300]),
            yaxis2=dict(title=dict(text='Air flow setpoint (%)', font=dict(color='blue')),
                        overlaying='y', side='right', range=[0, 120]),
            height=450, template='plotly_white', hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    
    with chart_col2:
        st.subheader("AVG Reactivity vs Target Reactivity")
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=[90] * len(filtered_df),
                                  fill=None, mode='lines', line=dict(width=0), showlegend=False, yaxis='y1'))
        fig5.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=[0] * len(filtered_df),
                                  fill='tonexty', mode='lines', line=dict(width=0),
                                  fillcolor='rgba(144, 238, 144, 0.5)', name='Target Reactivity t<90 sec', yaxis='y1'))
        fig5.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['3 Pt Average Reactivity (sec)'],
                                  name='3 Pt Average Reactivity (sec)', line=dict(color='purple', width=2), yaxis='y1'))
        
        fig5.update_layout(
            xaxis=dict(title='DATE/TIME'),
            yaxis=dict(title=dict(text='3 Pt Average Reactivity (seconds)', font=dict(color='purple')),
                       side='left', range=[0, 200]),
            yaxis2=dict(title=dict(text='Target Reactivity (Seconds)', font=dict(color='green')),
                        overlaying='y', side='right', range=[0, 200]),
            height=450, template='plotly_white', hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
        )
        
        st.plotly_chart(fig5, use_container_width=True)
    
    st.divider()
    
    # CHART 6: Gas Setting vs Gas Consumption
    st.header("Gas Setting vs Gas Consumption")
    
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['3 Pt Average Gas setting (% Openning)'],
                              name='3 Pt Average Gas setting (% Openning)', line=dict(color='blue', width=2), yaxis='y1'))
    fig6.add_trace(go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['3 Pt Average Gas Consumption (m^3)'],
                              name='3 Pt Average Gas Consumption (m^3)', line=dict(color='orange', width=2), yaxis='y2'))
    
    fig6.update_layout(
        xaxis=dict(title='DATE/TIME'),
        yaxis=dict(title=dict(text='Gas Openning Setpoint (%)', font=dict(color='blue')),
                   side='left', range=[0, 30]),
        yaxis2=dict(title=dict(text='Gas consumption (m^3/hour)', font=dict(color='orange')),
                    overlaying='y', side='right', range=[0, 300]),
        height=500, template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5)
    )
    
    st.plotly_chart(fig6, use_container_width=True)
    st.divider()
    
    # Temperature Deviation and Distribution
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        st.subheader("📊 Temperature Deviation from Target")
        
        deviation_fig = go.Figure()
        deviation = filtered_df['3 Pt Average Sinterting Temp'] - filtered_df['Desired Temp']
        colors = ['#FF4444' if x > 0 else '#4169E1' for x in deviation]
        
        deviation_fig.add_trace(go.Bar(x=filtered_df['Timestamp'], y=deviation, name="Deviation", marker_color=colors))
        deviation_fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)
        
        deviation_fig.update_layout(
            title="Temperature Deviation from Target (°C)", xaxis_title="Date & Time",
            yaxis_title="Deviation (°C)", height=450, template="plotly_white", showlegend=False
        )
        
        st.plotly_chart(deviation_fig, use_container_width=True)
    
    with chart_col4:
        st.subheader("📈 Temperature Distribution")
        
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=filtered_df['3 Pt Average Sinterting Temp'], name="Sintering Temp",
                                        nbinsx=30, marker_color='#FF4444', opacity=0.7))
        hist_fig.add_trace(go.Histogram(x=filtered_df['3PT Average Burner Temp'], name="Burner Temp",
                                        nbinsx=30, marker_color='#FF8C00', opacity=0.6))
        hist_fig.add_trace(go.Histogram(x=filtered_df['3 Pt Average Inlet Temp'], name="Inlet Temp",
                                        nbinsx=30, marker_color='#32CD32', opacity=0.5))
        
        hist_fig.update_layout(
            title="Temperature Distribution Across Zones", xaxis_title="Temperature (°C)",
            yaxis_title="Frequency", height=450, template="plotly_white",
            barmode='overlay', legend=dict(x=0.7, y=0.95)
        )
        
        st.plotly_chart(hist_fig, use_container_width=True)

# ============================================================================
# PAGE 2: ML ANALYTICS
# ============================================================================

elif page == "🤖 ML Analytics":
    st.markdown(f"""
        <div class="main-header">
            <h1>🤖 Machine Learning Analytics</h1>
            <p>AI-Powered Insights & Predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        "🎯 LOI Predictor",
        "⚠️ Anomaly Detection", 
        "⚡ Gas Optimization",
        "✅ Quality Classification"
    ])
    
    # TAB 1: LOI Predictor
    with ml_tab1:
        st.header("🎯 LOI Prediction Model")
        st.write("Predict LOI based on current operating conditions")
        
        if loi_model is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Model Performance")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Mean Absolute Error", f"{loi_metrics['MAE']:.3f}%")
                with metric_col2:
                    st.metric("RMSE", f"{loi_metrics['RMSE']:.3f}%")
                with metric_col3:
                    st.metric("R² Score", f"{loi_metrics['R2']:.3f}")
                
                df_pred = filtered_df[loi_features].dropna()
                if len(df_pred) > 0:
                    predictions = loi_model.predict(df_pred)
                    
                    fig_loi = go.Figure()
                    
                    fig_loi.add_trace(go.Scatter(
                        x=filtered_df.loc[df_pred.index, 'Timestamp'],
                        y=filtered_df.loc[df_pred.index, '3 Pt Average LOI (%)'],
                        name='Actual LOI',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_loi.add_trace(go.Scatter(
                        x=filtered_df.loc[df_pred.index, 'Timestamp'],
                        y=predictions,
                        name='Predicted LOI',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig_loi.add_hline(y=5, line_dash="dash", line_color="green",
                                     annotation_text="Target < 5%", annotation_position="right")
                    
                    fig_loi.update_layout(
                        title="Actual vs Predicted LOI",
                        xaxis_title="Date/Time",
                        yaxis_title="LOI (%)",
                        height=400,
                        template="plotly_white",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_loi, use_container_width=True)
            
            with col2:
                st.subheader("🎛️ What-If Analysis")
                st.write("Adjust parameters to see predicted LOI:")
                
                latest = filtered_df.iloc[-1]
                
                temp_input = st.slider("Sintering Temp (°C)", 
                                      float(df['3 Pt Average Sinterting Temp'].min()),
                                      float(df['3 Pt Average Sinterting Temp'].max()),
                                      float(latest['3 Pt Average Sinterting Temp']))
                
                gas_input = st.slider("Gas Setting (%)",
                                     float(df['3 Pt Average Gas setting (% Openning)'].min()),
                                     float(df['3 Pt Average Gas setting (% Openning)'].max()),
                                     float(latest['3 Pt Average Gas setting (% Openning)']))
                
                vsd_input = st.slider("VSD Speed (%)",
                                     float(df['VSD speed'].min()),
                                     float(df['VSD speed'].max()),
                                     float(latest['VSD speed']))
                
                input_data = pd.DataFrame({
                    '3 Pt Average Sinterting Temp': [temp_input],
                    '3PT Average Burner Temp': [latest['3PT Average Burner Temp']],
                    '3 Pt Average Inlet Temp': [latest['3 Pt Average Inlet Temp']],
                    '3 Pt Average Gas setting (% Openning)': [gas_input],
                    '3 Pt Average Gas Consumption (m^3)': [latest['3 Pt Average Gas Consumption (m^3)']],
                    'VSD speed': [vsd_input],
                    'Air Flow (%)': [latest['Air Flow (%)']]
                })
                
                predicted_loi = loi_model.predict(input_data)[0]
                
                st.metric("Predicted LOI", f"{predicted_loi:.2f}%",
                         delta="Good" if predicted_loi < 5 else "High",
                         delta_color="normal" if predicted_loi < 5 else "inverse")
        else:
            st.error("Not enough data to train LOI prediction model")
    
    # TAB 2: Anomaly Detection
    with ml_tab2:
        st.header("⚠️ Anomaly Detection")
        st.write("Identify unusual operating conditions that may lead to quality issues")
        
        if anomaly_model is not None and anomaly_scaler is not None:
            anomaly_features = [
                '3 Pt Average Sinterting Temp',
                '3PT Average Burner Temp',
                '3 Pt Average Inlet Temp',
                '3 Pt Average LOI (%)',
                '3 Pt Average Gas Consumption (m^3)',
                'VSD speed',
                'Air Flow (%)'
            ]
            
            df_anomaly = filtered_df[anomaly_features].dropna()
            
            if len(df_anomaly) > 0:
                X_scaled = anomaly_scaler.transform(df_anomaly)
                anomalies = anomaly_model.predict(X_scaled)
                anomaly_scores = anomaly_model.score_samples(X_scaled)
                
                filtered_df_anomaly = filtered_df.loc[df_anomaly.index].copy()
                filtered_df_anomaly['Anomaly'] = anomalies
                filtered_df_anomaly['Anomaly_Score'] = anomaly_scores
                
                n_anomalies = (anomalies == -1).sum()
                anomaly_pct = (n_anomalies / len(anomalies)) * 100
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    fig_anom = go.Figure()
                    
                    normal_mask = filtered_df_anomaly['Anomaly'] == 1
                    fig_anom.add_trace(go.Scatter(
                        x=filtered_df_anomaly.loc[normal_mask, 'Timestamp'],
                        y=filtered_df_anomaly.loc[normal_mask, '3 Pt Average LOI (%)'],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='green', size=6)
                    ))
                    
                    anomaly_mask = filtered_df_anomaly['Anomaly'] == -1
                    fig_anom.add_trace(go.Scatter(
                        x=filtered_df_anomaly.loc[anomaly_mask, 'Timestamp'],
                        y=filtered_df_anomaly.loc[anomaly_mask, '3 Pt Average LOI (%)'],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='red', size=10, symbol='x')
                    ))
                    
                    fig_anom.update_layout(
                        title="Anomaly Detection - LOI Over Time",
                        xaxis_title="Date/Time",
                        yaxis_title="LOI (%)",
                        height=400,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig_anom, use_container_width=True)
                
                with col2:
                    st.metric("Total Anomalies", f"{n_anomalies}")
                    st.metric("Anomaly Rate", f"{anomaly_pct:.1f}%")
                    
                    if n_anomalies > 0:
                        st.warning(f"⚠️ {n_anomalies} anomalies detected!")
                        
                        st.subheader("Recent Anomalies")
                        recent_anomalies = filtered_df_anomaly[filtered_df_anomaly['Anomaly'] == -1].tail(5)
                        
                        for idx, row in recent_anomalies.iterrows():
                            st.text(f"{row['Timestamp'].strftime('%Y-%m-%d %H:%M')}")
                            st.text(f"  LOI: {row['3 Pt Average LOI (%)']:.2f}%")
                            st.text(f"  Score: {row['Anomaly_Score']:.3f}")
                            st.divider()
        else:
            st.error("Not enough data to train anomaly detection model")
    
    # TAB 3: Gas Optimization
    with ml_tab3:
        st.header("⚡ Gas Consumption Optimization")
        st.write("Find optimal settings to minimize gas consumption while maintaining quality")
        
        if gas_model is not None and loi_model is not None:
            st.subheader("🎯 Optimization Results")
            
            current_gas = filtered_df['3 Pt Average Gas Consumption (m^3)'].mean()
            current_loi = filtered_df['3 Pt Average LOI (%)'].mean()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Avg Gas", f"{current_gas:.2f} m³")
            with col2:
                st.metric("Current Avg LOI", f"{current_loi:.2f}%")
            with col3:
                target_loi = st.number_input("Target LOI (%)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
            
            st.divider()
            
            st.subheader("💡 Recommended Settings")
            
            if current_loi > target_loi:
                st.info("**Recommendation:** LOI is above target. Consider:")
                st.write("- ✅ Increase sintering temperature by 10-20°C")
                st.write("- ✅ Reduce gas setting by 1-2%")
                st.write("- ✅ Adjust VSD speed for longer residence time")
            else:
                st.success("**Status:** LOI is within target range!")
                st.write("- ✅ Consider reducing gas setting by 0.5-1% to save energy")
                st.write("- ✅ Monitor closely to maintain quality")
            
            st.subheader("📊 Key Factors Affecting Gas Consumption")
            
            importance = gas_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': gas_features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance for Gas Consumption'
            )
            fig_importance.update_layout(height=400, template="plotly_white")
            
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.error("Not enough data to train optimization models")
    
    # TAB 4: Quality Classification
    with ml_tab4:
        st.header("✅ Quality Prediction")
        st.write("Predict if current conditions will produce good quality product (LOI < 5% AND Reactivity < 90 sec)")
        
        if quality_model is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Model Performance")
                
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric("Accuracy", f"{quality_metrics['Accuracy']*100:.1f}%")
                
                with metric_col2:
                    y_test = quality_metrics['y_test']
                    y_pred = quality_metrics['y_pred']
                    
                    if sum(y_pred) > 0:
                        precision = sum((y_test == 1) & (y_pred == 1)) / sum(y_pred)
                        st.metric("Precision (Good Quality)", f"{precision*100:.1f}%")
                
                cm = confusion_matrix(y_test, y_pred)
                
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Bad', 'Predicted Good'],
                    y=['Actual Bad', 'Actual Good'],
                    colorscale='RdYlGn',
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16}
                ))
                
                fig_cm.update_layout(
                    title="Confusion Matrix",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                st.subheader("🔮 Real-time Prediction")
                
                df_quality = filtered_df[quality_features].dropna()
                
                if len(df_quality) > 0:
                    quality_predictions = quality_model.predict(df_quality)
                    quality_proba = quality_model.predict_proba(df_quality)
                    
                    latest_pred = quality_predictions[-1]
                    latest_proba = quality_proba[-1]
                    
                    if latest_pred == 1:
                        st.success("✅ GOOD QUALITY PREDICTED")
                        st.metric("Confidence", f"{latest_proba[1]*100:.1f}%")
                    else:
                        st.error("❌ POOR QUALITY PREDICTED")
                        st.metric("Confidence", f"{latest_proba[0]*100:.1f}%")
                    
                    st.subheader("📈 Quality Trend")
                    quality_pct = (quality_predictions.sum() / len(quality_predictions)) * 100
                    st.metric("Good Quality Rate", f"{quality_pct:.1f}%")
                    
                    fig_quality = go.Figure()
                    
                    fig_quality.add_trace(go.Scatter(
                        x=filtered_df.loc[df_quality.index, 'Timestamp'],
                        y=quality_proba[:, 1],
                        mode='lines',
                        name='Good Quality Probability',
                        line=dict(color='green', width=2),
                        fill='tozeroy'
                    ))
                    
                    fig_quality.add_hline(y=0.5, line_dash="dash", line_color="red",
                                         annotation_text="Decision Threshold")
                    
                    fig_quality.update_layout(
                        title="Quality Probability Over Time",
                        xaxis_title="Date/Time",
                        yaxis_title="Probability",
                        height=300,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig_quality, use_container_width=True)
        else:
            st.error("Not enough data to train quality classification model")

# ============================================================================
# PAGE 3: PROCESS CAPABILITY (Cp/Cpk)
# ============================================================================

elif page == "📈 Process Capability (Cp/Cpk)":
    st.markdown(f"""
        <div class="main-header">
            <h1>📈 Process Capability Analysis</h1>
            <p>Cp, Cpk, and Process Reliability Metrics</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Select parameter to analyze
    st.subheader("🎯 Select Parameter for Analysis")
    
    param_options = {
        'LOI (%)': {
            'data': '3 Pt Average LOI (%)',
            'lsl': 0,
            'usl': 5,
            'target': 2.5,
            'unit': '%'
        },
        'Reactivity (sec)': {
            'data': '3 Pt Average Reactivity (sec)',
            'lsl': 0,
            'usl': 90,
            'target': 45,
            'unit': 'sec'
        },
        'Sintering Temperature (°C)': {
            'data': '3 Pt Average Sinterting Temp',
            'lsl': 900,
            'usl': 1000,
            'target': 950,
            'unit': '°C'
        },
        'Gas Consumption (m³)': {
            'data': '3 Pt Average Gas Consumption (m^3)',
            'lsl': 100,
            'usl': 250,
            'target': 175,
            'unit': 'm³'
        }
    }
    
    selected_param = st.selectbox("Select Parameter", list(param_options.keys()))
    
    param_config = param_options[selected_param]
    data_column = param_config['data']
    lsl = param_config['lsl']
    usl = param_config['usl']
    target = param_config['target']
    unit = param_config['unit']
    
    # Allow user to adjust limits
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lsl = st.number_input("Lower Specification Limit (LSL)", value=float(lsl), step=1.0)
    with col2:
        usl = st.number_input("Upper Specification Limit (USL)", value=float(usl), step=1.0)
    with col3:
        target = st.number_input("Target Value", value=float(target), step=1.0)
    
    st.divider()
    
    # Get data
    param_data = filtered_df[data_column].dropna().values
    
    if len(param_data) > 0:
        # Calculate Cp and Cpk
        cp, cpk, cpu, cpl = calculate_cp_cpk(param_data, lsl, usl, target)
        
        if cp is not None:
            # Display metrics
            st.subheader("📊 Process Capability Indices")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cp_color = "🟢" if cp >= 1.33 else "🟡" if cp >= 1.0 else "🔴"
                st.metric("Cp (Process Capability)", f"{cp:.3f} {cp_color}")
                st.caption("Cp ≥ 1.33: Capable | 1.0-1.33: Acceptable | < 1.0: Not Capable")
            
            with col2:
                cpk_color = "🟢" if cpk >= 1.33 else "🟡" if cpk >= 1.0 else "🔴"
                st.metric("Cpk (Process Capability Index)", f"{cpk:.3f} {cpk_color}")
                st.caption("Cpk ≥ 1.33: Capable | 1.0-1.33: Acceptable | < 1.0: Not Capable")
            
            with col3:
                st.metric("Cpu (Upper Capability)", f"{cpu:.3f}")
                st.caption("Distance from mean to USL")
            
            with col4:
                st.metric("Cpl (Lower Capability)", f"{cpl:.3f}")
                st.caption("Distance from mean to LSL")
            
            st.divider()
            
            # Process Performance Summary
            st.subheader("📋 Process Performance Summary")
            
            mean_val = np.mean(param_data)
            std_val = np.std(param_data, ddof=1)
            min_val = np.min(param_data)
            max_val = np.max(param_data)
            
            # Calculate % within spec
            within_spec = np.sum((param_data >= lsl) & (param_data <= usl))
            pct_within_spec = (within_spec / len(param_data)) * 100
            
            # Expected defect rate based on Cpk
            if cpk > 0:
                z_score = cpk * 3
                defect_rate = (1 - stats.norm.cdf(z_score)) * 2 * 1000000
            else:
                defect_rate = 1000000
            
            col1, col2 = st.columns(2)
            
            with col1:
                summary_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Range', 'Target'],
                    'Value': [
                        f"{mean_val:.2f} {unit}",
                        f"{std_val:.2f} {unit}",
                        f"{min_val:.2f} {unit}",
                        f"{max_val:.2f} {unit}",
                        f"{max_val - min_val:.2f} {unit}",
                        f"{target:.2f} {unit}"
                    ]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            with col2:
                performance_df = pd.DataFrame({
                    'Metric': ['% Within Spec', 'Expected Defects (PPM)', 'Sigma Level'],
                    'Value': [
                        f"{pct_within_spec:.2f}%",
                        f"{defect_rate:.0f}",
                        f"{cpk * 3:.2f}σ"
                    ]
                })
                st.dataframe(performance_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Capability Analysis Chart
            st.subheader("📈 Process Capability Histogram")
            
            fig_cap = go.Figure()
            
            # Histogram
            fig_cap.add_trace(go.Histogram(
                x=param_data,
                nbinsx=30,
                name='Data Distribution',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # Add specification limits
            fig_cap.add_vline(x=lsl, line_dash="dash", line_color="red", line_width=2,
                             annotation_text="LSL", annotation_position="top")
            fig_cap.add_vline(x=usl, line_dash="dash", line_color="red", line_width=2,
                             annotation_text="USL", annotation_position="top")
            fig_cap.add_vline(x=target, line_dash="solid", line_color="green", line_width=2,
                             annotation_text="Target", annotation_position="top")
            fig_cap.add_vline(x=mean_val, line_dash="dot", line_color="blue", line_width=2,
                             annotation_text="Mean", annotation_position="bottom")
            
            # Add normal distribution curve
            x_range = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 100)
            y_range = stats.norm.pdf(x_range, mean_val, std_val)
            
            # Scale to match histogram
            hist_max = len(param_data) / 30  # Approximate max height
            y_range_scaled = y_range * hist_max / np.max(y_range)
            
            fig_cap.add_trace(go.Scatter(
                x=x_range,
                y=y_range_scaled,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='darkblue', width=2)
            ))
            
            fig_cap.update_layout(
                title=f"Process Capability Analysis - {selected_param}",
                xaxis_title=f"{selected_param} ({unit})",
                yaxis_title="Frequency",
                height=500,
                template="plotly_white",
                showlegend=True
            )
            
            st.plotly_chart(fig_cap, use_container_width=True)
            
            st.divider()
            
            # Interpretation Guide
            st.subheader("📖 Interpretation Guide")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Cp (Process Capability):**")
                st.write("- **Cp ≥ 2.0**: Excellent - Process is highly capable")
                st.write("- **Cp ≥ 1.33**: Good - Process is capable")
                st.write("- **Cp ≥ 1.0**: Fair - Process is marginally capable")
                st.write("- **Cp < 1.0**: Poor - Process is not capable")
            
            with col2:
                st.markdown("**Cpk (Process Capability Index):**")
                st.write("- Measures actual process centering")
                st.write("- **Cpk = Cp**: Process is perfectly centered")
                st.write("- **Cpk < Cp**: Process is off-center")
                st.write("- Use Cpk for decision making")
            
            # Recommendations
            st.divider()
            st.subheader("💡 Recommendations")
            
            if cpk >= 1.33:
                st.success("✅ **Process is capable!** Continue monitoring to maintain performance.")
            elif cpk >= 1.0:
                st.warning("⚠️ **Process is marginally capable.** Consider:")
                st.write("- Reduce process variation (improve Cp)")
                st.write("- Center the process better (improve Cpk)")
                st.write("- Implement tighter process controls")
            else:
                st.error("❌ **Process is not capable.** Immediate action required:")
                st.write("- Identify and eliminate sources of variation")
                st.write("- Re-center the process")
                st.write("- Review and adjust specification limits if appropriate")
                st.write("- Implement statistical process control")
        else:
            st.error("Unable to calculate process capability (standard deviation is zero)")
    else:
        st.warning("No data available for selected parameter")

# ============================================================================
# PAGE 4: CONTROL CHARTS
# ============================================================================

elif page == "📉 Control Charts":
    st.markdown(f"""
        <div class="main-header">
            <h1>📉 Statistical Process Control Charts</h1>
            <p>X-bar & R Charts | I-MR Charts</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Select chart type
    chart_type = st.radio("Select Control Chart Type", 
                          ["X-bar & R Charts", "I-MR (Individual-Moving Range) Charts"],
                          horizontal=True)
    
    # Select parameter
    param_options_spc = {
        'LOI (%)': '3 Pt Average LOI (%)',
        'Reactivity (sec)': '3 Pt Average Reactivity (sec)',
        'Sintering Temperature (°C)': '3 Pt Average Sinterting Temp',
        'Gas Consumption (m³)': '3 Pt Average Gas Consumption (m^3)',
        'VSD Speed (%)': 'VSD speed'
    }
    
    selected_param_spc = st.selectbox("Select Parameter to Monitor", list(param_options_spc.keys()))
    data_column_spc = param_options_spc[selected_param_spc]
    
    st.divider()
    
    # Get data
    param_data_spc = filtered_df[data_column_spc].dropna().values
    timestamps_spc = filtered_df.loc[filtered_df[data_column_spc].notna(), 'Timestamp'].values
    
    if len(param_data_spc) > 0:
        
        # ====================================================================
        # X-BAR & R CHARTS
        # ====================================================================
        if chart_type == "X-bar & R Charts":
            st.header("📊 X-bar & R Control Charts")
            st.write("Monitor process mean (X-bar) and variability (Range)")
            
            # Subgroup size selection
            subgroup_size = st.slider("Subgroup Size", min_value=2, max_value=10, value=5, step=1)
            
            st.divider()
            
            # Calculate control limits
            control_limits = calculate_control_limits_xbar_r(param_data_spc, subgroup_size)
            
            # Create subgroup indices for x-axis
            n_subgroups = len(control_limits['xbar_values'])
            subgroup_indices = list(range(1, n_subgroups + 1))
            
            # X-bar Chart
            st.subheader("📈 X-bar Chart (Process Mean)")
            
            fig_xbar = go.Figure()
            
            # Plot X-bar values
            fig_xbar.add_trace(go.Scatter(
                x=subgroup_indices,
                y=control_limits['xbar_values'],
                mode='lines+markers',
                name='X-bar',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            # Add centerline
            fig_xbar.add_hline(
                y=control_limits['xbar_centerline'],
                line_dash="solid",
                line_color="green",
                line_width=2,
                annotation_text=f"CL = {control_limits['xbar_centerline']:.2f}",
                annotation_position="right"
            )
            
            # Add control limits
            fig_xbar.add_hline(
                y=control_limits['xbar_ucl'],
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"UCL = {control_limits['xbar_ucl']:.2f}",
                annotation_position="right"
            )
            
            fig_xbar.add_hline(
                y=control_limits['xbar_lcl'],
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"LCL = {control_limits['xbar_lcl']:.2f}",
                annotation_position="right"
            )
            
            # Highlight out-of-control points
            out_of_control_xbar = [
                i for i, val in enumerate(control_limits['xbar_values'])
                if val > control_limits['xbar_ucl'] or val < control_limits['xbar_lcl']
            ]
            
            if out_of_control_xbar:
                fig_xbar.add_trace(go.Scatter(
                    x=[subgroup_indices[i] for i in out_of_control_xbar],
                    y=[control_limits['xbar_values'][i] for i in out_of_control_xbar],
                    mode='markers',
                    name='Out of Control',
                    marker=dict(size=12, color='red', symbol='x')
                ))
            
            fig_xbar.update_layout(
                xaxis_title="Subgroup Number",
                yaxis_title=f"{selected_param_spc}",
                height=400,
                template="plotly_white",
                hovermode='x'
            )
            
            st.plotly_chart(fig_xbar, use_container_width=True)
            
            # R Chart
            st.subheader("📉 R Chart (Process Variability)")
            
            fig_r = go.Figure()
            
            # Plot R values
            fig_r.add_trace(go.Scatter(
                x=subgroup_indices,
                y=control_limits['r_values'],
                mode='lines+markers',
                name='Range',
                line=dict(color='orange', width=2),
                marker=dict(size=8)
            ))
            
            # Add centerline
            fig_r.add_hline(
                y=control_limits['r_centerline'],
                line_dash="solid",
                line_color="green",
                line_width=2,
                annotation_text=f"CL = {control_limits['r_centerline']:.2f}",
                annotation_position="right"
            )
            
            # Add control limits
            fig_r.add_hline(
                y=control_limits['r_ucl'],
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"UCL = {control_limits['r_ucl']:.2f}",
                annotation_position="right"
            )
            
            if control_limits['r_lcl'] > 0:
                fig_r.add_hline(
                    y=control_limits['r_lcl'],
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"LCL = {control_limits['r_lcl']:.2f}",
                    annotation_position="right"
                )
            
            # Highlight out-of-control points
            out_of_control_r = [
                i for i, val in enumerate(control_limits['r_values'])
                if val > control_limits['r_ucl'] or val < control_limits['r_lcl']
            ]
            
            if out_of_control_r:
                fig_r.add_trace(go.Scatter(
                    x=[subgroup_indices[i] for i in out_of_control_r],
                    y=[control_limits['r_values'][i] for i in out_of_control_r],
                    mode='markers',
                    name='Out of Control',
                    marker=dict(size=12, color='red', symbol='x')
                ))
            
            fig_r.update_layout(
                xaxis_title="Subgroup Number",
                yaxis_title="Range",
                height=400,
                template="plotly_white",
                hovermode='x'
            )
            
            st.plotly_chart(fig_r, use_container_width=True)
            
            # Control Chart Summary
            st.divider()
            st.subheader("📋 Control Chart Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Out of Control Points (X-bar)", len(out_of_control_xbar))
                st.metric("Process Centerline", f"{control_limits['xbar_centerline']:.2f}")
            
            with col2:
                st.metric("Out of Control Points (R)", len(out_of_control_r))
                st.metric("Average Range", f"{control_limits['r_centerline']:.2f}")
            
            if len(out_of_control_xbar) > 0 or len(out_of_control_r) > 0:
                st.warning("⚠️ **Process is out of control!** Investigate special causes of variation.")
            else:
                st.success("✅ **Process is in statistical control.** Continue monitoring.")
        
        # ====================================================================
        # I-MR CHARTS
        # ====================================================================
        else:  # I-MR Charts
            st.header("📊 I-MR Control Charts")
            st.write("Monitor individual measurements and moving ranges")
            
            st.divider()
            
            # Calculate control limits
            imr_limits = calculate_control_limits_imr(param_data_spc)
            
            # I Chart (Individuals)
            st.subheader("📈 I Chart (Individual Measurements)")
            
            fig_i = go.Figure()
            
            # Plot individual values
            fig_i.add_trace(go.Scatter(
                x=list(range(1, len(param_data_spc) + 1)),
                y=param_data_spc,
                mode='lines+markers',
                name='Individual Values',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # Add centerline
            fig_i.add_hline(
                y=imr_limits['x_bar'],
                line_dash="solid",
                line_color="green",
                line_width=2,
                annotation_text=f"CL = {imr_limits['x_bar']:.2f}",
                annotation_position="right"
            )
            
            # Add control limits
            fig_i.add_hline(
                y=imr_limits['i_ucl'],
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"UCL = {imr_limits['i_ucl']:.2f}",
                annotation_position="right"
            )
            
            fig_i.add_hline(
                y=imr_limits['i_lcl'],
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"LCL = {imr_limits['i_lcl']:.2f}",
                annotation_position="right"
            )
            
            # Highlight out-of-control points
            out_of_control_i = [
                i for i, val in enumerate(param_data_spc)
                if val > imr_limits['i_ucl'] or val < imr_limits['i_lcl']
            ]
            
            if out_of_control_i:
                fig_i.add_trace(go.Scatter(
                    x=[i+1 for i in out_of_control_i],
                    y=[param_data_spc[i] for i in out_of_control_i],
                    mode='markers',
                    name='Out of Control',
                    marker=dict(size=12, color='red', symbol='x')
                ))
            
            fig_i.update_layout(
                xaxis_title="Observation Number",
                yaxis_title=f"{selected_param_spc}",
                height=400,
                template="plotly_white",
                hovermode='x'
            )
            
            st.plotly_chart(fig_i, use_container_width=True)
            
            # MR Chart (Moving Range)
            st.subheader("📉 MR Chart (Moving Range)")
            
            fig_mr = go.Figure()
            
            # Plot moving range values
            fig_mr.add_trace(go.Scatter(
                x=list(range(2, len(param_data_spc) + 1)),
                y=imr_limits['moving_ranges'],
                mode='lines+markers',
                name='Moving Range',
                line=dict(color='orange', width=2),
                marker=dict(size=6)
            ))
            
            # Add centerline
            fig_mr.add_hline(
                y=imr_limits['mr_bar'],
                line_dash="solid",
                line_color="green",
                line_width=2,
                annotation_text=f"CL = {imr_limits['mr_bar']:.2f}",
                annotation_position="right"
            )
            
            # Add control limits
            fig_mr.add_hline(
                y=imr_limits['mr_ucl'],
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"UCL = {imr_limits['mr_ucl']:.2f}",
                annotation_position="right"
            )
            
            if imr_limits['mr_lcl'] > 0:
                fig_mr.add_hline(
                    y=imr_limits['mr_lcl'],
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"LCL = {imr_limits['mr_lcl']:.2f}",
                    annotation_position="right"
                )
            
            # Highlight out-of-control points
            out_of_control_mr = [
                i for i, val in enumerate(imr_limits['moving_ranges'])
                if val > imr_limits['mr_ucl'] or val < imr_limits['mr_lcl']
            ]
            
            if out_of_control_mr:
                fig_mr.add_trace(go.Scatter(
                    x=[i+2 for i in out_of_control_mr],
                    y=[imr_limits['moving_ranges'][i] for i in out_of_control_mr],
                    mode='markers',
                    name='Out of Control',
                    marker=dict(size=12, color='red', symbol='x')
                ))
            
            fig_mr.update_layout(
                xaxis_title="Observation Number",
                yaxis_title="Moving Range",
                height=400,
                template="plotly_white",
                hovermode='x'
            )
            
            st.plotly_chart(fig_mr, use_container_width=True)
            
            # Control Chart Summary
            st.divider()
            st.subheader("📋 Control Chart Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Out of Control Points (I)", len(out_of_control_i))
                st.metric("Process Mean", f"{imr_limits['x_bar']:.2f}")
            
            with col2:
                st.metric("Out of Control Points (MR)", len(out_of_control_mr))
                st.metric("Average Moving Range", f"{imr_limits['mr_bar']:.2f}")
            
            if len(out_of_control_i) > 0 or len(out_of_control_mr) > 0:
                st.warning("⚠️ **Process is out of control!** Investigate special causes of variation.")
            else:
                st.success("✅ **Process is in statistical control.** Continue monitoring.")
    else:
        st.warning("No data available for selected parameter")

# ============================================================================
# PAGE 5: ML MODEL DETAILS
# ============================================================================

elif page == "🔬 ML Model Details":
    st.markdown(f"""
        <div class="main-header">
            <h1>🔬 Machine Learning Model Details</h1>
            <p>Technical Specifications & Performance Metrics</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Model overview
    st.header("📋 Model Overview")
    
    models_info = pd.DataFrame({
        'Model': ['LOI Predictor', 'Gas Predictor', 'Quality Classifier', 'Anomaly Detector'],
        'Algorithm': ['Random Forest Regressor', 'Random Forest Regressor', 'Random Forest Classifier', 'Isolation Forest'],
        'Purpose': [
            'Predict LOI from operating conditions',
            'Predict gas consumption',
            'Classify quality as good/bad',
            'Detect unusual operating conditions'
        ],
        'Status': [
            '✅ Active' if loi_model else '❌ Inactive',
            '✅ Active' if gas_model else '❌ Inactive',
            '✅ Active' if quality_model else '❌ Inactive',
            '✅ Active' if anomaly_model else '❌ Inactive'
        ]
    })
    
    st.dataframe(models_info, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Feature importance for each model
    st.header("📊 Feature Importance Analysis")
    
    tab1, tab2, tab3 = st.tabs(["LOI Model", "Gas Model", "Quality Model"])
    
    with tab1:
        if loi_model is not None:
            importance = loi_model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': loi_features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                        title='Feature Importance - LOI Prediction Model')
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Model Metrics:**")
            st.write(f"- MAE: {loi_metrics['MAE']:.4f}")
            st.write(f"- RMSE: {loi_metrics['RMSE']:.4f}")
            st.write(f"- R² Score: {loi_metrics['R2']:.4f}")
    
    with tab2:
        if gas_model is not None:
            importance = gas_model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': gas_features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                        title='Feature Importance - Gas Consumption Model')
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Model Metrics:**")
            st.write(f"- MAE: {gas_metrics['MAE']:.4f}")
            st.write(f"- RMSE: {gas_metrics['RMSE']:.4f}")
            st.write(f"- R² Score: {gas_metrics['R2']:.4f}")
    
    with tab3:
        if quality_model is not None:
            importance = quality_model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': quality_features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                        title='Feature Importance - Quality Classification Model')
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Model Metrics:**")
            st.write(f"- Accuracy: {quality_metrics['Accuracy']*100:.2f}%")

# Footer
st.divider()
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; color: {TLOWANA_GRAY}; padding: 20px;">
        <p style="margin: 0;"><strong>TLOWANA RESOURCES</strong> | Combustion Area Dashboard</p>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">
            Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Showing {len(filtered_df):,} of {len(df):,} total records
        </p>
    </div>
""", unsafe_allow_html=True)
