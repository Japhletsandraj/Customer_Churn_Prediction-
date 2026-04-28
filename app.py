import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for professional styling with dark/light mode support
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Light mode (default) */
    :root {
        --bg-primary: #f5f7fa;
        --bg-secondary: #e4e8ec;
        --text-primary: #1a1a2e;
        --text-secondary: #666666;
        --card-bg: #ffffff;
        --card-shadow: rgba(0,0,0,0.08);
        --border-color: #e0e0e0;
    }
    
    /* Dark mode */
    [data-theme="dark"], .stApp[data-theme="dark"] {
        --bg-primary: #0e1117;
        --bg-secondary: #1a1a2e;
        --text-primary: #f0f0f0;
        --text-secondary: #a0a0a0;
        --card-bg: #262730;
        --card-shadow: rgba(0,0,0,0.3);
        --border-color: #3a3a4a;
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }
    
    .stApp {
        background: transparent;
    }
    
    h1, h2, h3 {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .stHeader {
        background: transparent;
    }
    
    /* Custom card styling */
    .custom-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px var(--card-shadow);
        margin-bottom: 20px;
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid var(--border-color);
    }
    
    .stNumberInput > div > div {
        border-radius: 12px;
        border: 2px solid var(--border-color);
    }
    
    .stSlider > div > div {
        border-radius: 8px;
    }
    
    /* Metric styling */
    div[data-testid="stMetric"] {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 12px var(--card-shadow);
    }
    
    div[data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: 14px;
    }
    
    div[data-testid="stMetricValue"] {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 12px;
        font-weight: 500;
        padding: 12px 24px;
    }
    
    /* Text colors for dynamic content */
    .dynamic-text {
        color: var(--text-primary);
    }
    
    .dynamic-text-secondary {
        color: var(--text-secondary);
    }
    </style>

    <script>
    // Detect system color scheme and apply theme
    (function() {
        const observer = new MutationObserver(function() {
            const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            const theme = isDark ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', theme);
        });
        
        // Initial check
        const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
        
        // Listen for changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
                document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
            });
        }
    })();
    </script>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_and_train_model():
    """Load data, preprocess, and train the ML model"""
    data = pd.read_csv("dataset/customer_churn_data.csv")
    
    # Preprocessing (same as notebook)
    data.drop(columns=["CustomerID"], inplace=True)
    data = data.replace({'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0})
    
    # Contract type encoding
    contract_score = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    data['contract_score'] = data['ContractType'].map(contract_score)
    
    # Risk score
    data['risk_score'] = 0
    data['risk_score'] += (data['MonthlyCharges'] > 70).astype(int)
    data['risk_score'] += (data['ContractType'] == 'Month-to-month').astype(int)
    data['risk_score'] += (data['Tenure'] < 12).astype(int)
    
    # Features for churn prediction
    features = ['MonthlyCharges', 'contract_score', 'Tenure', 'risk_score']
    X = data[features]
    y = data['Churn']
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    return rf, data, features

# Load model
rf_model, data, feature_names = load_and_train_model()

# Prediction function
def predict_churn(contract_type, tenure, monthly_charges):
    """Predict churn probability using the trained model"""
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    contract_score = contract_map[contract_type]
    
    risk_score = 0
    if monthly_charges > 70:
        risk_score += 1
    if contract_type == 'Month-to-month':
        risk_score += 1
    if tenure < 12:
        risk_score += 1
    
    features = np.array([[monthly_charges, contract_score, tenure, risk_score]])
    prob = rf_model.predict_proba(features)[0][1]
    
    return prob

# Header
st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 8px;">Customer Churn Prediction</h1>
        <p style="color: var(--text-secondary); font-size: 1.1rem;">AI-Powered Pricing & Retention Analysis</p>
    </div>
""", unsafe_allow_html=True)

# Main content
col_inputs, col_results = st.columns([1, 1], gap="large")

with col_inputs:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-bottom: 20px;">Customer Parameters</h3>', unsafe_allow_html=True)
    
    contract_type = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
        index=0
    )
    
    tenure = st.slider(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        help="Customer tenure in months"
    )
    
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0,
        max_value=200,
        value=70,
        step=1
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Get prediction
churn_probability = predict_churn(contract_type, tenure, monthly_charges)
churn_prediction = "Yes" if churn_probability > 0.5 else "No"

with col_results:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-bottom: 20px;">Prediction Results</h3>', unsafe_allow_html=True)
    
    # Main prediction display
    if churn_probability > 0.5:
        pred_color = "#e74c3c"
        pred_text = "High Risk"
    elif churn_probability > 0.3:
        pred_color = "#f39c12"
        pred_text = "Medium Risk"
    else:
        pred_color = "#27ae60"
        pred_text = "Low Risk"
    
    # Custom prediction gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = churn_probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability", 'font': {'size': 16, 'color': '#1a1a2e'}},
        delta = {'reference': 50, 'increasing': {'color': "#e74c3c"}, 'decreasing': {'color': "#27ae60"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': pred_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, 30], 'color': "#d5f5e3"},
                {'range': [30, 50], 'color': "#fef9e7"},
                {'range': [50, 100], 'color': "#fadbd8"}
            ],
            'threshold': {
                'line': {'color': "#333", 'width': 2},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig_gauge.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    
    # Risk level badge
    st.markdown(f"""
        <div style="text-align: center; padding: 12px; background: {pred_color}15; 
                    border-radius: 12px; margin-top: 10px;">
            <span style="font-size: 1.2rem; font-weight: 600; color: {pred_color};">{pred_text}</span>
            <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 8px 0 0 0;">
                {churn_probability:.1%} probability of churn
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Visualizations section
st.markdown("---")

col_viz1, col_viz2 = st.columns([1, 1], gap="large")

with col_viz1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-bottom: 15px;">Churn by Contract Type</h3>', unsafe_allow_html=True)
    
    # Contract type analysis
    contract_churn = data.groupby('ContractType')['Churn'].mean().reset_index()
    contract_churn['Churn_Percent'] = contract_churn['Churn'] * 100
    
    fig_contract = px.bar(
        contract_churn, 
        x='ContractType', 
        y='Churn_Percent',
        color='Churn_Percent',
        color_continuous_scale=['#27ae60', '#f39c12', '#e74c3c'],
        range_color=[0, 50],
        text_auto='.1f'
    )
    fig_contract.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title="",
        yaxis_title="Churn Rate (%)",
        yaxis=dict(range=[0, 50]),
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        height=280
    )
    fig_contract.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    
    st.plotly_chart(fig_contract, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_viz2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-bottom: 15px;">Monthly Charges vs Churn</h3>', unsafe_allow_html=True)
    
    # Create charge bins
    data['ChargeBin'] = pd.cut(data['MonthlyCharges'], bins=[0, 50, 60, 70, 80, 90, 200], 
                                labels=['<50', '50-60', '60-70', '70-80', '80-90', '90+'])
    charge_churn = data.groupby('ChargeBin')['Churn'].mean().reset_index()
    charge_churn['Churn_Percent'] = charge_churn['Churn'] * 100
    
    fig_charges = px.bar(
        charge_churn,
        x='ChargeBin',
        y='Churn_Percent',
        color='Churn_Percent',
        color_continuous_scale=['#27ae60', '#f39c12', '#e74c3c'],
        range_color=[0, 50],
        text_auto='.1f'
    )
    fig_charges.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title="Monthly Charges ($)",
        yaxis_title="Churn Rate (%)",
        yaxis=dict(range=[0, 50]),
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        height=280
    )
    fig_charges.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    
    st.plotly_chart(fig_charges, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# Tenure analysis
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown('<h3 style="margin-bottom: 15px;">Churn Rate by Tenure</h3>', unsafe_allow_html=True)

data['TenureBin'] = pd.cut(data['Tenure'], bins=[-1, 6, 12, 24, 48, 100], 
                           labels=['0-6', '7-12', '13-24', '25-48', '49+'])
tenure_churn = data.groupby('TenureBin')['Churn'].mean().reset_index()
tenure_churn['Churn_Percent'] = tenure_churn['Churn'] * 100

fig_tenure = px.line(
    tenure_churn,
    x='TenureBin',
    y='Churn_Percent',
    markers=True,
    line_shape='spline'
)
fig_tenure.update_traces(
    line_color='#3498db',
    marker=dict(size=10, color='#3498db', line=dict(color='white', width=2))
)
fig_tenure.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis_title="Tenure (months)",
    yaxis_title="Churn Rate (%)",
    yaxis=dict(range=[0, max(tenure_churn['Churn_Percent']) + 10]),
    showlegend=False,
    margin=dict(l=20, r=20, t=30, b=20),
    height=300
)

st.plotly_chart(fig_tenure, use_container_width=True, config={'displayModeBar': False})
st.markdown('</div>', unsafe_allow_html=True)

# Recommendations section
st.markdown("---")
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown('<h3 style="margin-bottom: 20px;">Strategic Recommendations</h3>', unsafe_allow_html=True)

rec_col1, rec_col2, rec_col3 = st.columns(3)

with rec_col1:
    if contract_type == "Month-to-month":
        rec_text = "Offer 12-24 month contracts with discounts to reduce churn"
        rec_color = "#e74c3c"
    elif contract_type == "One year":
        rec_text = "Encourage upgrade to 2-year contract for better retention"
        rec_color = "#f39c12"
    else:
        rec_text = "Customer is in stable contract - focus on satisfaction"
        rec_color = "#27ae60"
    
    st.markdown(f"""
        <div style="padding: 16px; background: {rec_color}10; border-radius: 12px; border-left: 4px solid {rec_color};">
            <h4 style="margin: 0 0 8px 0; color: {rec_color};">Contract Strategy</h4>
            <p style="margin: 0; color: var(--text-primary); font-size: 0.95rem;">{rec_text}</p>
        </div>
    """, unsafe_allow_html=True)

with rec_col2:
    if monthly_charges > 70:
        rec_text = f"Consider reducing to ${max(62, monthly_charges - 10):.0f}-${monthly_charges - 5:.0f} to lower churn risk"
        rec_color = "#e74c3c"
    elif monthly_charges > 60:
        rec_text = "Current pricing is reasonable - monitor for adjustments"
        rec_color = "#f39c12"
    else:
        rec_text = "Pricing is optimal - maintain current strategy"
        rec_color = "#27ae60"
    
    st.markdown(f"""
        <div style="padding: 16px; background: {rec_color}10; border-radius: 12px; border-left: 4px solid {rec_color};">
            <h4 style="margin: 0 0 8px 0; color: {rec_color};">Pricing Recommendation</h4>
            <p style="margin: 0; color: var(--text-primary); font-size: 0.95rem;">{rec_text}</p>
        </div>
    """, unsafe_allow_html=True)

with rec_col3:
    if tenure < 12:
        rec_text = "Focus on early engagement - first 12 months are critical"
        rec_color = "#e74c3c"
    elif tenure < 24:
        rec_text = "Customer is maturing - offer loyalty rewards"
        rec_color = "#f39c12"
    else:
        rec_text = "Long-term customer - prioritize retention programs"
        rec_color = "#27ae60"
    
    st.markdown(f"""
        <div style="padding: 16px; background: {rec_color}10; border-radius: 12px; border-left: 4px solid {rec_color};">
            <h4 style="margin: 0 0 8px 0; color: {rec_color};">Tenure Action</h4>
            <p style="margin: 0; color: var(--text-primary); font-size: 0.95rem;">{rec_text}</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px; color: var(--text-secondary); font-size: 0.85rem;">
        <p>Powered by Random Forest Machine Learning Model</p>
    </div>
""", unsafe_allow_html=True)