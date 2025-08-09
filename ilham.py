import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

# Set page config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2E86C1;
        border-bottom: 2px solid #2E86C1;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-card {
        background-color: #27AE60;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .error-card {
        background-color: #E74C3C;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .warning-card {
        background-color: #F39C12;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86C1;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
CHART_COLORS = {
    'primary': '#2E86C1',
    'secondary': '#E74C3C',
    'success': '#27AE60',
    'warning': '#F39C12',
    'info': '#17A2B8'
}

API_BASE_URL = "http://localhost:8502"

# Helper functions
def create_main_header(text):
    return f'<div class="main-header">{text}</div>'

def create_section_header(text):
    return f'<div class="section-header">{text}</div>'

def create_metric_card(value, label, color="gradient"):
    if color == "gradient":
        card_class = "metric-card"
    elif color == "success":
        card_class = "success-card"
    elif color == "error":
        card_class = "error-card"
    elif color == "warning":
        card_class = "warning-card"
    else:
        card_class = "metric-card"
    
    return f'<div class="{card_class}"><h3>{value}</h3><p>{label}</p></div>'

def create_info_box(content):
    return f'<div class="info-box">{content}</div>'

# Load data
@st.cache_resource
def clear_cache():
    cache_path = os.path.join(os.getcwd(), ".streamlit", "cache")
    if os.path.exists(cache_path):
        for f in os.listdir(cache_path):
            os.remove(os.path.join(cache_path, f))
        st.success("Cache dibersihkan!")
    else:
        st.warning("Folder cache tidak ditemukan")

# 2. Modifikasi fungsi load_data dengan error handling
@st.cache_data(persist="disk", show_spinner="Memuat data...")
def load_data():
    try:
        # Ganti dengan path file yang benar
        data_path = "train.csv" 
        
        # Baca dengan parameter tambahan untuk handle formatting
        data = pd.read_csv(
            data_path,
            engine='python',
            encoding='utf-8',
            on_bad_lines='skip'
        )
        
        # Konversi index ke string untuk menghindari masalah serialisasi
        data.index = data.index.astype(str)
        
        return data
    
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return pd.DataFrame()  # Return dataframe kosong jika error

# 3. Tambahkan tombol clear cache di sidebar
with st.sidebar:
    if st.button("🔄 Bersihkan Cache"):
        clear_cache()
        st.experimental_rerun()

# 4. Load data dengan error handling
try:
    data = load_data()
    if data.empty:
        st.warning("Data kosong atau gagal dimuat. Periksa file input.")
        st.stop()
except Exception as e:
    st.error(f"Error saat memuat data: {str(e)}")
    st.stop()

# 5. Tambahkan validasi data
required_columns = ['SalePrice', 'OverallQual', 'GrLivArea']
missing_cols = [col for col in required_columns if col not in data.columns]

if missing_cols:
    st.error(f"Kolom penting tidak ditemukan: {missing_cols}")
    st.stop()

# Initialize session state for profile
if 'profile' not in st.session_state:
    st.session_state.profile = {
        'name': '',
        'email': '',
        'age': 30,
        'occupation': '',
        'interest': '',
        'profile_complete': False
    }

# Sidebar for navigation
st.sidebar.title("Navigation")
options =st.sidebar.radio("Select a page:", 
                          ["👨 maker profile", "Home", "Data Exploration", "Price Prediction", "My Profile", "About"])

 #PAGE 1: PROFILE
if options == "👨 maker profile":
    st.markdown(create_main_header("Ilham Muharya"), unsafe_allow_html=True)
    st.markdown(create_section_header("Data & AI Specialist"), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(create_info_box("""
        <h3>📧 Contact Information</h3>
        <p><strong>Email:</strong> ilhammuharya@gmail.com</p>
        <p><strong>Phone:</strong> +62 815 1724 9427</p>
        <p><strong>Location:</strong> Tangerang Metropolitan Area</p>
        <p><strong>LinkedIn:</strong> www.linkedin.com/in/ilham-muharya-ilham/</p>
        <p><strong>GitHub:</strong> github.com/ilhammuharya-lab</p>
        """), unsafe_allow_html=True)
    
    st.markdown(create_section_header("Professional Summary"), unsafe_allow_html=True)
    st.write("""
    I’m an Industrial Engineering graduate currently working as a Production Foreman at PT Surya TOTO Indonesia. With hands-on experience in managing manufacturing operations and leading teams on the production floor, I’ve developed strong communication and problem-solving skills that help me navigate challenges efficiently.

    Currently, I’m diving deeper into the world of Machine Learning and programming, particularly with Python, because I believe the future of manufacturing lies in the integration of intelligent systems and data-driven decision-making.

        🧠 Key strengths:
    • Effective Communication
    • Critical Thinking & Analytical Skills
    • Python Programming
    • Continuous Learning Attitude

    I'm excited to connect with professionals who share a passion for smart manufacturing, data, and future-ready technologies. Let’s grow together.
    """)
    
   
    
    # Skills
    st.markdown(create_section_header("Technical Skills"), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Programming & ML**
        - Python, SQL, R, PySpark
        - Scikit-learn, TensorFlow, PyTorch
        - LangChain, OpenAI
        
        **Data Visualization**
        - Tableau, Plotly
        - Streamlit, Dash
        """)
    
    with col2:
        st.markdown("""
        **MLOps & Cloud**
        - Docker
        - FastAPI
        
        **Databases & Tools**
        - PostgreSQL, MongoDB
        - Git, GitHub, VS Code
        """)

# Home page
if options == "Home":
    st.title("🏠 House Price Prediction App")
    st.markdown("""
    Welcome to the House Price Prediction App! This application helps you:
    - Explore housing data
    - Predict house prices based on various features
    - Understand the factors that influence house prices
    
    Use the navigation panel on the left to explore different sections.
    """)
    
    st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
             use_column_width=True)
    
    st.subheader("Dataset Overview")
    st.write(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
    st.write("Here are the first few rows of the dataset:")
    st.dataframe(data.head())

# Data Exploration page
elif options == "Data Exploration":
    st.title("🔍 Data Exploration")
    
    st.subheader("Dataset Summary")
    st.write(data.describe())
    
    st.subheader("Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribution of Sale Prices**")
        fig, ax = plt.subplots()
        sns.histplot(data['SalePrice'], kde=True, ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.write("**Correlation with Sale Price**")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr = data[numeric_cols].corr()['SalePrice'].sort_values(ascending=False)[1:11]
        fig, ax = plt.subplots()
        sns.barplot(x=corr.values, y=corr.index, ax=ax)
        st.pyplot(fig)
    
    st.subheader("Filter Data")
    selected_cols = st.multiselect("Select columns to display:", data.columns)
    if selected_cols:
        st.dataframe(data[selected_cols])

# Price Prediction page
elif options == "Price Prediction":
    st.title("📈 Price Prediction")
    
    st.markdown("""
    This section allows you to predict house prices based on selected features.
    The model uses a Random Forest Regressor trained on the dataset.
    """)
    
    # Feature selection
    st.subheader("Select Features for Prediction")
    
    # Important features based on correlation
    important_features = [
        'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
        'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
        'YearBuilt', 'YearRemodAdd'
    ]
    
    selected_features = st.multiselect(
        "Select features to use for prediction:", 
        data.columns.drop('SalePrice'),
        default=important_features
    )
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        st.stop()
    
    # Prepare data
    X = data[selected_features]
    y = data['SalePrice']
    
    # Handle missing values - simple imputation for demo
    for col in X.select_dtypes(include=[np.number]):
        X[col].fillna(X[col].median(), inplace=True)
    
    for col in X.select_dtypes(exclude=[np.number]):
        X[col].fillna(X[col].mode()[0], inplace=True)
    
    # Convert categorical variables
    X = pd.get_dummies(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Model Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    
    # Plot actual vs predicted
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Actual vs Predicted Prices")
    st.pyplot(fig)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=importance.head(10), ax=ax)
    st.pyplot(fig)
    
    # Prediction interface
    st.subheader("Make a Prediction")
    st.write("Enter values for the selected features to get a price prediction:")
    
    input_data = {}
    cols = st.columns(2)
    for i, feature in enumerate(selected_features):
        col = cols[i % 2]
        
        if data[feature].dtype == 'object':
            options = data[feature].unique()
            input_data[feature] = col.selectbox(feature, options)
        else:
            min_val = float(data[feature].min())
            max_val = float(data[feature].max())
            default_val = float(data[feature].median())
            input_data[feature] = col.number_input(
                feature, min_val, max_val, default_val
            )
    
    if st.button("Predict Price"):
        # Prepare input for prediction
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)
        
        # Ensure all columns from training are present
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[X.columns]
        
        # Make prediction
        prediction = model.predict(input_df)
        
        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

# Profile page
elif options == "My Profile":
    st.title("👤 My Profile")
    
    with st.form("profile_form"):
        st.subheader("Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", value=st.session_state.profile['name'])
            email = st.text_input("Email Address", value=st.session_state.profile['email'])
            age = st.slider("Age", 18, 100, value=st.session_state.profile['age'])
        
        with col2:
            occupation = st.selectbox(
                "Occupation",
                ["", "Student", "Engineer", "Doctor", "Teacher", "Business", "Other"],
                index=0 if not st.session_state.profile['occupation'] else 
                ["", "Student", "Engineer", "Doctor", "Teacher", "Business", "Other"].index(st.session_state.profile['occupation'])
            )
            interest = st.selectbox(
                "Primary Interest in Housing",
                ["", "Buying", "Renting", "Investing", "Research"],
                index=0 if not st.session_state.profile['interest'] else 
                ["", "Buying", "Renting", "Investing", "Research"].index(st.session_state.profile['interest'])
            )
            today = date.today()
            member_since = st.date_input("Member Since", value=today)
        
        # Profile picture upload
        profile_pic = st.file_uploader("Upload Profile Picture", type=["jpg", "png", "jpeg"])
        
        submitted = st.form_submit_button("Save Profile")
        
        if submitted:
            if not name or not email:
                st.error("Please fill in required fields (Name and Email)")
            else:
                st.session_state.profile = {
                    'name': name,
                    'email': email,
                    'age': age,
                    'occupation': occupation,
                    'interest': interest,
                    'member_since': member_since,
                    'profile_complete': True
                }
                if profile_pic is not None:
                    st.session_state.profile['profile_pic'] = profile_pic
                st.success("Profile saved successfully!")
    
    # Display profile information
    if st.session_state.profile['profile_complete']:
        st.subheader("Your Profile Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'profile_pic' in st.session_state.profile and st.session_state.profile['profile_pic'] is not None:
                st.image(st.session_state.profile['profile_pic'], width=200)
            else:
                st.image("https://cdn.pixabay.com/photo/2015/10/05/22/37/blank-profile-picture-973460_1280.png", width=200)
            
            st.write(f"**Name:** {st.session_state.profile['name']}")
            st.write(f"**Email:** {st.session_state.profile['email']}")
            st.write(f"**Age:** {st.session_state.profile['age']}")
        
        with col2:
            st.write(f"**Occupation:** {st.session_state.profile['occupation']}")
            st.write(f"**Interest:** {st.session_state.profile['interest']}")
            st.write(f"**Member Since:** {st.session_state.profile['member_since']}")
            
            # Display some stats based on profile
            if st.session_state.profile['interest'] == "Buying":
                st.info("Based on your interest in buying, you might want to check our mortgage calculator.")
            elif st.session_state.profile['interest'] == "Investing":
                st.info("As an investor, you might be interested in our rental yield calculator.")

# About page
elif options == "About":
    st.title("ℹ️ About")
    
    st.markdown("""
    ### House Price Prediction App
    
    This application is designed to:
    - Explore housing market data
    - Predict house prices based on various features
    - Help users understand factors influencing house prices
    
    **Dataset Information:**
    - The dataset contains information about residential homes
    - Includes features like property size, quality, age, and more
    
    **Technologies Used:**
    - Python
    - Streamlit for the web interface
    - Scikit-learn for machine learning
    - Pandas for data manipulation
    - Matplotlib and Seaborn for visualization
    
    **How to Use:**
    1. Explore the data in the Data Exploration section
    2. Use the Price Prediction section to get estimates
    3. Complete your profile in the My Profile section
    4. Select features and input values to get predictions
    
    For questions or feedback, please contact the developer.
    """)

# Run the app
if __name__ == "__main__":
    pass
