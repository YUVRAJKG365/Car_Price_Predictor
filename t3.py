import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- SESSION STATE --------------------
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}

# -------------------- LOAD MODEL & DATA --------------------
@st.cache_resource
def load_model():
    try:
        with open("car_price_prediction_model.pkl", "rb") as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        st.error("🚨 Model file `car_price_prediction_model.pkl` not found!")
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv("car data.csv")
    except FileNotFoundError:
        st.error("🚨 Data file `car data.csv` not found!")
        return None

model = load_model()
df_original = load_data()

# -------------------- ADVANCED FUNCTIONS --------------------
def calculate_car_age(year):
    return datetime.now().year - year

def get_car_brand(car_name):
    return car_name.split()[0] if car_name else "Unknown"

def predict_price(input_features):
    """Make prediction with error handling"""
    try:
        categorical_cols = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
        df_encoded = pd.get_dummies(df_original, columns=categorical_cols, drop_first=True)
        df_encoded["Car_Age"] = datetime.now().year - df_encoded["Year"]
        model_columns = df_encoded.drop(columns=["Selling_Price", "Car_Name_ritz", "Year"]).columns

        input_encoded = pd.get_dummies(input_features, columns=categorical_cols, drop_first=True)
        input_encoded["Car_Age"] = datetime.now().year - input_encoded["Year"]
        final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(final_input)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def update_prediction_history(input_data, prediction):
    """Update prediction history"""
    timestamp = datetime.now()
    st.session_state.prediction_history.append({
        'timestamp': timestamp,
        'car_name': input_data['Car_Name'],
        'brand': get_car_brand(input_data['Car_Name']),
        'year': input_data['Year'],
        'present_price': input_data['Present_Price'],
        'kms_driven': input_data['Driven_kms'],
        'fuel_type': input_data['Fuel_Type'],
        'transmission': input_data['Transmission'],
        'predicted_price': prediction,
        'car_age': calculate_car_age(input_data['Year'])
    })

def calculate_depreciation(present_price, predicted_price, car_age):
    """Calculate depreciation metrics"""
    total_depreciation = present_price - predicted_price
    annual_depreciation = total_depreciation / car_age if car_age > 0 else 0
    depreciation_rate = (total_depreciation / present_price) * 100 if present_price > 0 else 0
    return total_depreciation, annual_depreciation, depreciation_rate

# -------------------- ENHANCED CSS --------------------
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #f0f4f9 0%, #e0f2fe 50%, #f0f4f9 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    h1, h2, h3 {
        color: #1e3a8a;
        font-weight: 700;
        text-align: center;
    }

    /* Main Container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #1e3a8a, #2563eb);
        color: white;
        border-radius: 40px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.3);
    }

    /* Input Styling */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTextInput input {
        border-radius: 12px;
        border: 2px solid #93c5fd;
        padding: 8px 12px;
        background-color: #f9fafb;
        transition: all 0.3s ease;
    }
    .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"]:focus, .stTextInput input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }

    /* Prediction Result */
    .result-box {
        padding: 2rem;
        border-radius: 20px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    .result-box:hover {
        transform: translateY(-5px);
    }
    .success {
        background: linear-gradient(135deg, #16a34a, #22c55e);
        color: white;
    }
    .info {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: white;
    }

    /* Stats Cards */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }

    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #2563eb;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Tabs */
    .stTabs [role="tablist"] {
        gap: 1rem;
        justify-content: center;
    }
    .stTabs [role="tab"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.5rem;
        border: 1px solid #e5e7eb;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- MAIN APP --------------------
def main():
    # Header Section
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🚗 Smart Car Price Predictor</h1>
        <h3 style="color: #4b5563; margin-top: 0;">by Yuvraj Kumar Gond</h3>
        <p style="font-size: 1.2rem; color: #374151; max-width: 800px; margin: 0 auto;">
            Advanced AI-powered car valuation with market insights, depreciation analysis, 
            and comprehensive price estimation 🚀
        </p>
    </div>
    """, unsafe_allow_html=True)

    if model is None or df_original is None:
        st.stop()

    # Quick Stats Overview
    if st.session_state.prediction_history:
        total_predictions = len(st.session_state.prediction_history)
        avg_predicted_price = np.mean([p['predicted_price'] for p in st.session_state.prediction_history])
        most_common_brand = max(set([p['brand'] for p in st.session_state.prediction_history]), 
                              key=[p['brand'] for p in st.session_state.prediction_history].count)
        
        col1, col2, col3, col4 = st.columns(4)
        card_height = "250px"  # Define a fixed height for the cards
        with col1:
            st.markdown(f"""
            <div class="stats-card" style="height: {card_height};">
            <h3>📊 Total</h3>
            <h2>{total_predictions}</h2>
            <p>Predictions Made</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stats-card" style="background: linear-gradient(135deg, #16a34a, #22c55e); height: {card_height};">
            <h3>💰 Average Price</h3>
            <h2>₹{avg_predicted_price:.1f}L</h2>
            <p>Mean Predicted Value</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stats-card" style="background: linear-gradient(135deg, #f59e0b, #eab308); height: {card_height};">
            <h3>🏆 Popular Brand</h3>
            <h2>{most_common_brand}</h2>
            <p>Most Analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            unique_brands = len(set([p['brand'] for p in st.session_state.prediction_history]))
            st.markdown(f"""
            <div class="stats-card" style="background: linear-gradient(135deg, #ec4899, #db2777); height: {card_height};">
            <h3>🎯 Brands</h3>
            <h2>{unique_brands}</h2>
            <p>Unique Brands</p>
            </div>
            """, unsafe_allow_html=True)

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Price Prediction", "📊 Market Insights", "📈 Analytics"])

    # ---- Tab 1: Price Prediction ----
    with tab1:
        
        st.subheader("🔧 Enter Car Details")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            car_name = st.selectbox(
                "🚗 Car Brand & Model",
                options=sorted(df_original["Car_Name"].unique()),
                help="Select your car's make and model"
            )
            year = st.number_input(
                "📅 Manufacturing Year",
                min_value=1980,
                max_value=datetime.now().year,
                value=2018,
                help="Year the car was manufactured"
            )
            present_price = st.number_input(
                "💰 Current Showroom Price (Lakhs)",
                min_value=1.0,
                max_value=100.0,
                value=8.5,
                step=0.1,
                help="Current ex-showroom price for a new model"
            )

        with col2:
            kms_driven = st.number_input(
                "🛣️ Kilometers Driven", 
                min_value=100, 
                value=45000, 
                step=500,
                help="Total distance the car has been driven"
            )
            fuel_type = st.selectbox(
                "⛽ Fuel Type", 
                options=df_original["Fuel_Type"].unique(),
                help="Type of fuel the car uses"
            )
            selling_type = st.selectbox(
                "🏪 Selling Type", 
                options=df_original["Selling_type"].unique(),
                help="Type of selling arrangement"
            )

        with col3:
            transmission = st.selectbox(
                "⚙️ Transmission", 
                options=df_original["Transmission"].unique(),
                help="Transmission type"
            )
            owner = st.number_input(
                "👥 Number of Previous Owners", 
                min_value=0, 
                max_value=10, 
                value=1,
                help="How many people owned the car before"
            )
            car_age = calculate_car_age(year)
            st.metric("🕒 Car Age", f"{car_age} years")

        # Prediction Button
        if st.button("🔮 Predict Selling Price", use_container_width=True, type="primary"):
            with st.spinner("Analyzing market trends and calculating optimal price... ⏳"):
                # Create input data
                input_dict = {
                    "Car_Name": car_name,
                    "Year": year,
                    "Present_Price": present_price,
                    "Driven_kms": kms_driven,
                    "Fuel_Type": fuel_type,
                    "Selling_type": selling_type,
                    "Transmission": transmission,
                    "Owner": owner,
                }
                
                prediction = predict_price(pd.DataFrame([input_dict]))
                
                if prediction is not None:
                    # Update history
                    update_prediction_history(input_dict, prediction)
                    
                    # Calculate depreciation
                    total_dep, annual_dep, dep_rate = calculate_depreciation(
                        present_price, prediction, car_age
                    )
                    
                    # Display Main Result
                    st.markdown("---")
                    st.subheader("🎯 Prediction Result")
                    
                    col_res1, col_res2 = st.columns([2, 1])
                    
                    with col_res1:
                        st.markdown(
                            f'<div class="result-box success">💰 Predicted Selling Price: <br><span style="font-size:2rem;">₹ {prediction:.2f} Lakhs</span></div>',
                            unsafe_allow_html=True
                        )
                    
                    with col_res2:
                        st.metric("Car Age", f"{car_age} years")
                        st.metric("Depreciation Rate", f"{dep_rate:.1f}%")
                    
                    # Depreciation Analysis
                    st.subheader("📉 Depreciation Analysis")
                    dep_col1, dep_col2, dep_col3, dep_col4 = st.columns(4)
                    
                    with dep_col1:
                        st.metric("Original Price", f"₹{present_price:.2f}L")
                    with dep_col2:
                        st.metric("Current Value", f"₹{prediction:.2f}L")
                    with dep_col3:
                        st.metric("Total Depreciation", f"₹{total_dep:.2f}L")
                    with dep_col4:
                        st.metric("Annual Depreciation", f"₹{annual_dep:.2f}L/yr")
                    
                    # Feature Impact Analysis
                    st.subheader("🔍 Feature Impact")
                    feat_col1, feat_col2, feat_col3 = st.columns(3)
                    
                    with feat_col1:
                        st.markdown(f"""
                        <div class="feature-card">
                            <strong>🚗 Vehicle Info</strong><br>
                            Brand: {get_car_brand(car_name)}<br>
                            Model: {car_name}<br>
                            Age: {car_age} years
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with feat_col2:
                        st.markdown(f"""
                        <div class="feature-card">
                            <strong>📊 Usage Metrics</strong><br>
                            KMs Driven: {kms_driven:,}<br>
                            Owners: {owner}<br>
                            Transmission: {transmission}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with feat_col3:
                        st.markdown(f"""
                        <div class="feature-card">
                            <strong>⛽ Fuel & Type</strong><br>
                            Fuel: {fuel_type}<br>
                            Selling Type: {selling_type}<br>
                            Condition: {'Good' if kms_driven < 50000 else 'Average'}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.balloons()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Tab 2: Market Insights ----
    with tab2:
        
        st.subheader("📊 Market Insights & Trends")
        
        if df_original is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Price distribution by fuel type
                fig_fuel = px.box(df_original, x='Fuel_Type', y='Selling_Price', 
                                title='💰 Price Distribution by Fuel Type',
                                color='Fuel_Type')
                st.plotly_chart(fig_fuel, use_container_width=True)
                
                # Transmission impact
                trans_stats = df_original.groupby('Transmission')['Selling_Price'].mean().reset_index()
                fig_trans = px.pie(trans_stats, values='Selling_Price', names='Transmission',
                                 title='⚙️ Average Price by Transmission')
                st.plotly_chart(fig_trans, use_container_width=True)
            
            with col2:
                # Year vs Price
                fig_year = px.scatter(df_original, x='Year', y='Selling_Price', 
                                    color='Fuel_Type', size='Present_Price',
                                    title='📅 Price Trend by Manufacturing Year',
                                    trendline="lowess")
                st.plotly_chart(fig_year, use_container_width=True)
                
                # Owner impact
                owner_stats = df_original.groupby('Owner')['Selling_Price'].mean().reset_index()
                fig_owner = px.bar(owner_stats, x='Owner', y='Selling_Price',
                                 title='👥 Impact of Previous Owners on Price',
                                 color='Selling_Price')
                st.plotly_chart(fig_owner, use_container_width=True)
        else:
            st.info("Market insights will be available once data is loaded.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Tab 3: Analytics ----
    with tab3:
        
        st.subheader("📈 Prediction Analytics")
        
        if st.session_state.prediction_history:
            # Convert history to DataFrame
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Price trend over predictions
                fig_trend = px.line(history_df, x=history_df.index, y='predicted_price',
                                  title='📊 Price Prediction Trend',
                                  markers=True)
                fig_trend.update_layout(xaxis_title='Prediction Number', 
                                      yaxis_title='Predicted Price (Lakhs)')
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Brand distribution
                brand_counts = history_df['brand'].value_counts().reset_index()
                brand_counts.columns = ['Brand', 'Count']
                fig_brands = px.pie(brand_counts, values='Count', names='Brand',
                                  title='🏆 Brand Distribution in Predictions')
                st.plotly_chart(fig_brands, use_container_width=True)
            
            with col2:
                # Age vs Price
                fig_age = px.scatter(history_df, x='car_age', y='predicted_price',
                                   color='fuel_type', size='present_price',
                                   title='🕒 Car Age vs Predicted Price',
                                   trendline="lowess")
                st.plotly_chart(fig_age, use_container_width=True)
                
                # Recent Predictions
                st.subheader("🕒 Recent Predictions")
                recent_data = history_df.tail(5)[['car_name', 'predicted_price', 'timestamp']]
                for _, row in recent_data.iterrows():
                    st.write(f"**{row['car_name']}** - ₹{row['predicted_price']:.2f}L")
            
            # Export Data
            st.subheader("📤 Export Data")
            if st.button("Download Prediction History", use_container_width=True):
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"car_price_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("📊 Analytics will appear here after you make some predictions. Go to the 'Price Prediction' tab to get started!")
        
        st.markdown('</div>', unsafe_allow_html=True)

    
    # -------------------- SIDEBAR --------------------
    with st.sidebar:
        st.header("📘 About the Project")
        st.info(
            "This advanced AI system predicts used car prices using **Machine Learning**. "
            "It analyzes market trends, depreciation patterns, and vehicle specifications "
            "to provide accurate price estimates."
        )
        
        st.header("🎯 How It Works")
        st.success("""
        1. **Input** car details
        2. **Analyze** market patterns
        3. **Calculate** depreciation
        4. **Predict** optimal price
        5. **Provide** insights
        """)
        
        st.header("📞 Support")
        st.warning("For accurate predictions:\n- Provide correct mileage\n- Include all features\n- Consider vehicle condition")

# -------------------- FOOTER --------------------
    st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #6b7280;">
    <hr style="border: 1px solid #e5e7eb; margin-bottom: 1rem;">
    <p>Built with ❤️ using Streamlit • Powered by Machine Learning • Smart Car Price Predictor v2.0</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()