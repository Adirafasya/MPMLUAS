import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model terbaik
model = joblib.load('best_model.pkl')

# Memuat data untuk pengkodean dan penskalaan
data = pd.read_csv('onlinefoods.csv')

# Daftar kolom yang diperlukan selama pelatihan
required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code']

# Pastikan hanya kolom yang diperlukan ada
data = data[required_columns]

# Pra-pemrosesan data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = data[column].astype(str)
    le.fit(data[column])
    data[column] = le.transform(data[column])
    label_encoders[column] = le

scaler = StandardScaler()
numeric_features = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Fungsi untuk memproses input pengguna
def preprocess_input(user_input):
    processed_input = {col: [user_input.get(col, 'Unknown')] for col in required_columns}
    for column in label_encoders:
        if column in processed_input:
            input_value = processed_input[column][0]
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])
            else:
                # Jika nilai tidak dikenal, berikan nilai default seperti -1
                processed_input[column] = [-1]
    processed_input = pd.DataFrame(processed_input)
    processed_input[numeric_features] = scaler.transform(processed_input[numeric_features])
    return processed_input

# CSS untuk gaya dengan warna baby pink dan biru tua
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600&display=swap');
    .main {
        background-color: #FDF2F8; /* Baby Pink */
        font-family: 'Baloo 2', cursive;
    }
    h1 {
        color: #003366; /* Dark Blue */
        text-align: center;
        margin-bottom: 25px;
        font-family: 'Baloo 2', cursive;
    }
    h3 {
        color: #003366; /* Dark Blue */
        font-family: 'Baloo 2', cursive;
    }
    .stButton>button {
        background-color: #003366; /* Dark Blue */
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-family: 'Baloo 2', cursive;
    }
    .stButton>button:hover {
        background-color: #001a33; /* Darker Blue */
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
    }
    .css-1offfwp {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Antarmuka Streamlit
st.image("https://via.placeholder.com/800x200.png?text=Prediksi+Feedback+Pelanggan+Online+Food", use_column_width=True)
st.title("Prediksi Feedback Pelanggan Online Food")

st.markdown("""
    <h3>Masukkan Data Pelanggan</h3>
""", unsafe_allow_html=True)

# Membagi tampilan menjadi dua kolom
col1, col2 = st.columns(2)

# Input pengguna di kolom kiri
with col1:
    st.markdown('<p style="color: #003366; font-size: 16px; font-family: \'Baloo 2\', cursive;">Age</p>', unsafe_allow_html=True)
    age = st.number_input('', min_value=18, max_value=100)
    
    st.markdown('<p style="color: #003366; font-size: 16px; font-family: \'Baloo 2\', cursive;">Gender</p>', unsafe_allow_html=True)
    gender = st.selectbox('', ['Male', 'Female'])
    
    st.markdown('<p style="color: #003366; font-size: 16px; font-family: \'Baloo 2\', cursive;">Marital Status</p>', unsafe_allow_html=True)
    marital_status = st.selectbox('', ['Single', 'Married'])
    
    st.markdown('<p style="color: #003366; font-size: 16px; font-family: \'Baloo 2\', cursive;">Occupation</p>', unsafe_allow_html=True)
    occupation = st.selectbox('', ['Student', 'Employee', 'Self Employed'])
    
    st.markdown('<p style="color: #003366; font-size: 16px; font-family: \'Baloo 2\', cursive;">Monthly Income</p>', unsafe_allow_html=True)
    monthly_income = st.selectbox('', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'])

# Input pengguna di kolom kanan
with col2:
    st.markdown('<p style="color: #003366; font-size: 16px; font-family: \'Baloo 2\', cursive;">Educational Qualifications</p>', unsafe_allow_html=True)
    educational_qualifications = st.selectbox('', ['Under Graduate', 'Graduate', 'Post Graduate'])
    
    st.markdown('<p style="color: #003366; font-size: 16px; font-family: \'Baloo 2\', cursive;">Family size</p>', unsafe_allow_html=True)
    family_size = st.number_input('', min_value=1, max_value=20)
    
    st.markdown('<p style="color: #003366; font-size: 16px; font-family: \'Baloo 2\', cursive;">Latitude</p>', unsafe_allow_html=True)
    latitude = st.number_input('', format="%f")
    
    st.markdown('<p style="color: #003366; font-size: 16px; font-family: \'Baloo 2\', cursive;">Longitude</p>', unsafe_allow_html=True)
    longitude = st.number_input('', format="%f")
    
    st.markdown('<p style="color: #003366; font-size: 16px; font-family: \'Baloo 2\', cursive;">Pin code</p>', unsafe_allow_html=True)
    pin_code = st.number_input('', min_value=100000, max_value=999999)

user_input = {
    'Age': age,
    'Gender': gender,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Monthly Income': monthly_income,
    'Educational Qualifications': educational_qualifications,
    'Family size': family_size,
    'latitude': latitude,
    'longitude': longitude,
    'Pin code': pin_code
}

if st.button('Predict'):
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        st.write(f'Prediction: {prediction[0]}')
    except ValueError as e:
        st.error(f"Error in prediction: {e}")

# Tambahkan elemen HTML untuk output
st.markdown("""
    <h3>Output Prediksi</h3>
    <p>Hasil prediksi akan ditampilkan di sini.</p>
""", unsafe_allow_html=True)
