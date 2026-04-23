import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Algorithmic Pricing Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CUSTOM CSS (SLICK DARK THEME) ---
st.markdown("""
<style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Text Colors */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Glassmorphism Metric Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 10px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    /* Color Specifics for Cards */
    .card-base { border-top: 4px solid #3b82f6; }
    .card-optimal { border-top: 4px solid #10b981; }
    .card-revenue { border-top: 4px solid #f59e0b; }
    
    .val-base { color: #60a5fa; }
    .val-optimal { color: #34d399; }
    .val-revenue { color: #fbbf24; }

    /* Clean up Streamlit elements */
    header {visibility: hidden;}
    .css-1544g2n.e1fqcg0o4 {padding-top: 2rem;}
    
    /* Divider Customization */
    hr {
        border-color: rgba(255,255,255,0.1);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        with open('model/pricing_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/features.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features
    except Exception as e:
        return None, None

@st.cache_data(show_spinner=False)
def load_data():
    try:
        return pd.read_csv('data/listings.csv')
    except Exception as e:
        return None

def main():
    model, features = load_model()
    listings = load_data()
    
    if model is None or listings is None:
        st.error("⚠️ Data or model not found. Please ensure `data_generator.py` and `train_model.py` have been run.")
        return
        
    # --- HEADER SECTION ---
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown("<h1 style='font-size: 2.5rem; font-weight: 700; margin-bottom: 0;'>⚡ Algorithmic Pricing Engine</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Dynamic revenue optimization powered by Machine Learning.</p>", unsafe_allow_html=True)
    with col_header2:
        st.info("💡 **Tip:** Adjust the context parameters in the sidebar to simulate different market conditions.")
    
    st.markdown("<hr>", unsafe_allow_html=True)
        
    # --- SIDEBAR (CONTROLS) ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=60) # Placeholder for a slick logo
        st.markdown("## Configuration Panel")
        
        # Property Selection
        with st.expander("🏠 1. Select Property", expanded=True):
            property_id = st.selectbox(
                "Choose a Listing", 
                listings['property_id'].tolist(),
                help="Select a property to analyze its specific demand curve."
            )
            
            prop_data = listings[listings['property_id'] == property_id].iloc[0]
            st.caption(f"**{prop_data['name']}**")
            
            # Mini specs
            c1, c2 = st.columns(2)
            c1.metric("Beds", prop_data['bedrooms'])
            c2.metric("Rating", f"{prop_data['rating']} ⭐")
            st.metric("Neighborhood", prop_data['neighborhood'])
            
        # Context Simulation
        with st.expander("🌤️ 2. Market Context", expanded=True):
            st.markdown("<small style='color:#94a3b8;'>Simulate conditions to see how the optimal price shifts.</small>", unsafe_allow_html=True)
            
            month = st.slider("Month of Year", 1, 12, 6, help="Seasonality significantly impacts willingness to pay.")
            
            # Interactive Toggles with better UI
            st.markdown("**Day Characteristics**")
            is_weekend = st.toggle("Is it a Weekend?", value=True)
            is_holiday = st.toggle("Is it a Holiday?", value=False)
            is_event = st.toggle("Local Event Nearby?", value=False)
            
            st.markdown("**Weather Forecast**")
            weather = st.selectbox(
                "Expected Weather", 
                ['Sunny', 'Cloudy', 'Rainy', 'Snowy'],
                help="Weather affects neighborhood desirability (e.g., Beachfront needs Sunny weather)."
            )

    # --- CORE PREDICTION LOGIC ---
    base_price = prop_data['base_price']
    min_price = max(20, int(base_price * 0.4))
    max_price = int(base_price * 2.8)
    test_prices = list(range(min_price, max_price + 10, 5))
    
    pred_data = []
    for price in test_prices:
        row = {
            'price': price,
            'is_weekend': 1 if is_weekend else 0,
            'is_holiday': 1 if is_holiday else 0,
            'is_event': 1 if is_event else 0,
            'bedrooms': prop_data['bedrooms'],
            'base_price': prop_data['base_price'],
            'rating': prop_data['rating'],
            'month': month,
            'weather_Cloudy': 1 if weather == 'Cloudy' else 0,
            'weather_Rainy': 1 if weather == 'Rainy' else 0,
            'weather_Snowy': 1 if weather == 'Snowy' else 0,
            'neighborhood_Arts District': 1 if prop_data['neighborhood'] == 'Arts District' else 0,
            'neighborhood_Beachfront': 1 if prop_data['neighborhood'] == 'Beachfront' else 0,
            'neighborhood_Downtown': 1 if prop_data['neighborhood'] == 'Downtown' else 0,
            'neighborhood_Suburbs': 1 if prop_data['neighborhood'] == 'Suburbs' else 0,
            'neighborhood_Uptown': 1 if prop_data['neighborhood'] == 'Uptown' else 0,
        }
        pred_data.append(row)
        
    df_pred = pd.DataFrame(pred_data)
    for f in features:
        if f not in df_pred.columns: df_pred[f] = 0
            
    X_predict = df_pred[features]
    probabilities = model.predict_proba(X_predict)[:, 1]
    expected_revenues = [p * prob for p, prob in zip(test_prices, probabilities)]
    
    max_rev_index = np.argmax(expected_revenues)
    optimal_price = test_prices[max_rev_index]
    max_expected_revenue = expected_revenues[max_rev_index]
    optimal_prob = probabilities[max_rev_index]
    
    # --- KPI DASHBOARD ---
    st.markdown("### Executive Summary")
    st.markdown("Based on the selected context, here is the algorithmic recommendation for pricing:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
        <div class="glass-card card-base">
            <div class="metric-label">Original Base Price</div>
            <div class="metric-value val-base">${base_price}</div>
        </div>
        ''', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f'''
        <div class="glass-card card-optimal">
            <div class="metric-label">AI Recommended Price</div>
            <div class="metric-value val-optimal">${optimal_price}</div>
        </div>
        ''', unsafe_allow_html=True)
        
    with col3:
        st.markdown(f'''
        <div class="glass-card card-revenue">
            <div class="metric-label">Max Expected Revenue</div>
            <div class="metric-value val-revenue">${max_expected_revenue:.2f}</div>
            <div style="font-size: 0.8rem; color:#94a3b8; margin-top:5px;">@ {optimal_prob*100:.1f}% Booking Probability</div>
        </div>
        ''', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
        
    # --- INTERACTIVE VISUALIZATIONS ---
    st.markdown("### Deep Dive Analysis")
    st.markdown("<small style='color:#94a3b8;'>Hover over the charts for detailed data points.</small>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📈 Elasticity Curves", "🧠 Model Explainability (SHAP proxy)"])
    
    with tab1:
        col_fig1, col_fig2 = st.columns(2)
        
        # Shared Layout attributes for slick look
        layout_args = dict(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e6ed', family='Inter'),
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # 1. Expected Revenue Curve
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=test_prices, y=expected_revenues, 
            mode='lines', name='Expected Revenue', 
            line=dict(color='#f59e0b', width=4, shape='spline'), 
            fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.1)',
            hovertemplate="Price: $%{x}<br>Revenue: $%{y:.2f}"
        ))
        fig2.add_vline(x=optimal_price, line_dash="dot", line_color="#10b981", annotation_text=f" Optimal: ${optimal_price}", annotation_font_color="#10b981")
        
        fig2.update_layout(
            title='Revenue Optimization Curve',
            xaxis_title='Offered Price ($)',
            yaxis_title='Expected Revenue ($)',
            **layout_args
        )
        fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        
        # 2. Demand Curve (Price vs Probability)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=test_prices, y=probabilities, 
            mode='lines', name='Booking Probability', 
            line=dict(color='#3b82f6', width=4, shape='spline'),
            hovertemplate="Price: $%{x}<br>Probability: %{y:.1%}"
        ))
        fig1.add_vline(x=optimal_price, line_dash="dot", line_color="#10b981", annotation_text=f" Optimal: ${optimal_price}", annotation_font_color="#10b981")
        
        fig1.update_layout(
            title='Demand Curve (Price Elasticity)',
            xaxis_title='Offered Price ($)',
            yaxis_title='Probability of Booking',
            yaxis=dict(range=[0, 1.05], tickformat='.0%'),
            **layout_args
        )
        fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        
        with col_fig1: st.plotly_chart(fig2, use_container_width=True)
        with col_fig2: st.plotly_chart(fig1, use_container_width=True)
            
    with tab2:
        st.markdown("**What factors are driving this property's demand?**")
        st.info("This chart extracts the Random Forest's internal feature importance, showing which variables it relies on most heavily to make predictions.")
        importances = model.feature_importances_
        
        clean_features = []
        for f in features:
            if f.startswith('weather_'): clean_features.append(f"Weather: {f.split('_')[1]}")
            elif f.startswith('neighborhood_'): clean_features.append(f"Area: {f.split('_')[1]}")
            else: clean_features.append(f.replace('_', ' ').title())
            
        feat_df = pd.DataFrame({'Feature': clean_features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        
        fig_feat = px.bar(
            feat_df.tail(8), 
            x='Importance', y='Feature', 
            orientation='h', 
            color='Importance',
            color_continuous_scale="Blues"
        )
        fig_feat.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e6ed', family='Inter'),
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig_feat.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig_feat, use_container_width=True)

if __name__ == "__main__":
    main()
