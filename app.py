import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.metrics import Precision, Recall

from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import io
import os
import random
import matplotlib.pyplot as plt
import json
from datetime import datetime

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AcneSolve: Deteksi & Rekomendasi Jerawat",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

np.set_printoptions(suppress=True)
tf.get_logger().setLevel('ERROR')

JOURNAL_DIR = "progress_journal_data"
JOURNAL_METADATA_FILE = os.path.join(JOURNAL_DIR, "journal_metadata.json")

def load_journal_entries():
    if not os.path.exists(JOURNAL_DIR):
        os.makedirs(JOURNAL_DIR)
    if os.path.exists(JOURNAL_METADATA_FILE):
        with open(JOURNAL_METADATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_journal_entries(entries):
    with open(JOURNAL_METADATA_FILE, 'w') as f:
        json.dump(entries, f, indent=4)

def add_journal_entry(image_bytes, note):
    """Saves image bytes to file and adds entry to journal metadata."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_pil = Image.open(io.BytesIO(image_bytes))
        image_filename = f"progress_image_{timestamp}.jpg"
        image_path = os.path.join(JOURNAL_DIR, image_filename)
        
        img_pil.save(image_path, format="JPEG")

        new_entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": image_filename, 
            "note": note if note else "Tidak ada catatan."
        }
        journal_entries.append(new_entry)
        save_journal_entries(journal_entries)
        return True, "Progres berhasil disimpan!"
    except Exception as e:
        return False, f"Gagal menyimpan progres: {e}"


journal_entries = load_journal_entries()

# --- Fungsi untuk Membuat Arsitektur Model EfficientNet (SAMA PERSIS DENGAN SAAT PELATIHAN, TANPA DATA AUGMENTATION) ---
def create_efficientnet_model_architecture(num_classes=5):
    efficientnet_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(150, 150, 3)
    )

    for layer in efficientnet_model.layers[:-50]:
        layer.trainable = True

    input_layer = layers.Input(shape=(150, 150, 3))
    x = efficientnet_model(input_layer, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# --- Fungsi untuk Memuat Model dan Data Rekomendasi ---
@st.cache_resource
def load_ml_assets():
    """Loads the Keras model (architecture + weights) and label file manually."""
    weights_only_path = r"D:\DOKUMEN MATKUL\SEMESTER 6\MACHINE LEARNING\jerawat\acne_predict_weight.h5" 
    labels_path = "labels.txt"

    if not os.path.exists(weights_only_path):
        st.error(f"❌ ERROR: Model weights file '{weights_only_path}' NOT FOUND.")
        st.info(f"Ensure '{weights_only_path}' is in the correct directory and is a WEIGHTS-ONLY .h5 file.")
        return None, None
    
    if not os.path.exists(labels_path):
        st.error(f"❌ ERROR: Labels file '{labels_path}' NOT FOUND.")
        st.info(f"Ensure '{labels_path}' is in the same directory as the app.")
        return None, None

    try:
        # Menghapus pesan loading di sidebar untuk tampilan yang lebih bersih
        # st.sidebar.info(f"Building model architecture and loading weights...")
        
        NUM_CLASSES = 5 
        model = create_efficientnet_model_architecture(num_classes=NUM_CLASSES)
        model.load_weights(weights_only_path)
        
        # st.sidebar.success("✅ Model and weights loaded successfully!")

        # st.sidebar.info(f"Loading acne type list...")
        with open(labels_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        # st.sidebar.success("✅ Acne type list loaded successfully!")

        if not class_names:
            st.error("❌ `labels.txt` is empty or does not contain class names. Please check the file.")
            return None, None
        
        # st.sidebar.write(f"Loaded classes: {class_names}") # Debugging message, dihapus
        # st.sidebar.write(f"TensorFlow Version: {tf.__version__}") # Debugging message, dihapus

        return model, class_names
    except Exception as e:
        st.error(f"❌ Failed to build/load model or weights: {e}")
        st.warning("Common causes: Mismatch in model architecture definition, corrupted weights file, or TensorFlow version issues. **Please check the console for full error traceback.**")
        st.info("""
        **Troubleshooting steps:**
        1.  **Architecture Mismatch (Most Common with this approach):** The `create_efficientnet_model_architecture` function MUST exactly match your trained model. Check every layer, filter, unit, input_shape, activations.
        2.  **Weights File Integrity:** Ensure `acne_predict_weight.h5` exists, is not corrupted, and truly contains *only weights*.
        3.  **TensorFlow Version Consistency:** Verify exact TF version consistency between training and Streamlit environments (`pip show tensorflow`).
        4.  **Dependencies:** Ensure `h5py` and `protobuf` versions are compatible with your TensorFlow.
        """)
        return None, None

@st.cache_data
def load_recommendation_data():
    """Loads recommendation data from CSV."""
    csv_path = "pengobatan.csv"

    if not os.path.exists(csv_path):
        st.error(f"❌ ERROR: Recommendation file '{csv_path}' NOT FOUND.")
        st.info(f"Ensure '{csv_path}' is in the same directory as this app.")
        return None

    try:
        # Menghapus pesan loading di sidebar untuk tampilan yang lebih bersih
        # st.sidebar.info(f"Loading recommendation data...")
        df = pd.read_csv(csv_path)
        # st.sidebar.success("✅ Recommendation data loaded successfully!")
        
        required_cols = ['tingkat', 'jenis kulit', 'kandungan', 'rekomendasi']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in 'pengobatan.csv'. Ensure all required columns are present and named exactly.")
        
        df['tingkat'] = df['tingkat'].astype(str).str.lower().str.replace('s$', '', regex=True)
        df['jenis kulit'] = df['jenis kulit'].astype(str).str.lower()
        
        return df
    except Exception as e:
        st.error(f"❌ Failed to load file '{csv_path}': {e}")
        st.warning("Common causes: file missing, incorrect CSV format (delimiter, missing columns), or unusual characters.")
        st.info("Ensure the file is in the same directory and the format is correct (columns: 'tingkat', 'jenis kulit', 'kandungan', 'rekomendasi').")
        return None

# Load all ML assets and data at app startup
model, class_names = load_ml_assets()
df_pengobatan = load_recommendation_data()

# Mapping for skin type from Streamlit to 'jenis kulit' column in CSV
SKIN_TYPE_MAP_TO_CSV = {
    "Normal": "semua",
    "Berminyak": "oily",
    "Kering": "semua",
    "Kombinasi": "semua",
    "Sensitive": "semua", 
    "Tidak Tahu / Lewati": "semua" 
}

# --- Interactive Features: "Did You Know?" ---
DID_YOU_KNOW_FACTS = [
    "Tahukah Anda? Mencuci muka dua kali sehari sudah cukup. Terlalu sering bisa mengiritasi kulit!",
    "Tahukah Anda? Stres dapat memicu jerawat. Kelola stres dengan baik untuk kulit yang lebih sehat.",
    "Tahukah Anda? Sarung bantal kotor bisa jadi sarang bakteri penyebab jerawat. Ganti secara rutin!",
    "Tahukah Anda? SPF itu wajib, bahkan untuk kulit berjerawat! Pilih yang non-comedogenic.",
    "Tahukah Anda? Jangan memencet jerawat! Ini bisa memperparah peradangan dan meninggalkan bekas luka.",
    "Tahukah Anda? Asupan gula berlebih bisa memperburuk jerawat. Kurangi makanan manis untuk kulit lebih baik.",
    "Tahukah Anda? Produk non-comedogenic berarti tidak akan menyumbat pori-pori.",
    "Tahukah Anda? Antibiotik oral untuk jerawat harus digunakan di bawah pengawasan dokter."
]

# --- Interactive Features: Myth vs. Fact ---
MYTH_FACT_PAIRS = [
    {"myth": "Mitos: Pasta gigi bisa mengeringkan jerawat.",
     "fact": "Fakta: Pasta gigi dapat mengiritasi dan memperburuk jerawat karena mengandung bahan-bahan seperti alkohol dan mentol yang tidak dirancang untuk kulit."},
    {"myth": "Mitos: Matahari bisa menyembuhkan jerawat.",
     "fact": "Fakta: Paparan sinar matahari berlebihan justru bisa merusak kulit, memicu produksi minyak berlebih, dan membuat bekas jerawat lebih gelap (hiperpigmentasi pasca-inflamasi)."},
    {"myth": "Mitos: Jerawat hanya dialami remaja.",
     "fact": "Fakta: Jerawat bisa muncul pada usia berapa pun, dari bayi hingga dewasa. Jerawat dewasa sering disebut 'adult acne'."},
    {"myth": "Mitos: Memencet jerawat akan mempercepat penyembuhan.",
     "fact": "Fakta: Memencet jerawat dapat mendorong bakteri lebih dalam ke kulit, menyebabkan infeksi, peradangan lebih parah, dan meninggalkan bekas luka atau flek hitam."},
    {"myth": "Mitos: Kulit kering tidak bisa berjerawat.",
     "fact": "Fakta: Kulit kering juga bisa berjerawat, terutama jika pori-pori tersumbat atau ada ketidakseimbangan mikrobioma kulit. Jerawat bisa terjadi pada semua jenis kulit."},
]


# --- CUSTOM CSS INJECTION (Macaroon Style) ---
st.markdown("""
<style>
    /* Macaroon Color Palette */
    :root {
        --macaroon-light-bg: #F7F3F3; /* Almost white */
        --macaroon-sidebar-bg: #E6DCDC; /* Light Grayish Pink */
        --macaroon-main-header: #6B4E56; /* Plum/Dark Muted Pink */
        --macaroon-subheader: #9B7B8B; /* Muted Rose */
        --macaroon-text: #4A4A4A; /* Soft Dark Gray */
        --macaroon-accent-pink: #D8A7B1; /* Rose Pink */
        --macaroon-accent-teal: #A8DADC; /* Soft Teal */
        --macaroon-success-light: #F0F4C3; /* Light Lime */
        --macaroon-success-dark: #A2C49C; /* Soft Green */
        --macaroon-info-light: #D5E8F3; /* Light Sky Blue */
        --macaroon-info-dark: #81B3D5; /* Muted Blue */
        --macaroon-warning-light: #FDEBD0; /* Creamy Yellow */
        --macaroon-warning-dark: #E6B0AA; /* Muted Orange */
        --macaroon-error-light: #FADBD8; /* Light Coral */
        --macaroon-error-dark: #EB9991; /* Darker Coral */
        --macaroon-border: #E0E0E0; /* Light Gray Border */
    }

    /* General Font and Background */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Poppins', sans-serif;
        background-color: var(--macaroon-light-bg);
        color: var(--macaroon-text);
    }
    .stApp {
        background-color: var(--macaroon-light-bg);
    }

    /* Main Header */
    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem !important;
        font-weight: 700;
        color: var(--macaroon-main-header);
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.15);
        padding-top: 1rem;
    }

    /* Subheaders */
    .subheader {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem !important;
        font-weight: 600;
        color: var(--macaroon-subheader);
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid var(--macaroon-border);
        padding-bottom: 0.8rem;
        text-align: left;
    }

    /* General Text */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stVerticalBlock"] label {
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
        line-height: 1.7;
        color: var(--macaroon-text);
    }

    /* Buttons */
    .stButton>button {
        background-color: var(--macaroon-accent-pink); /* Rose Pink */
        color: white;
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        cursor: pointer;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #C28E99; /* Darker pink on hover */
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }

    /* Radio Buttons, Selectbox, File Uploader Labels */
    .stRadio > label, .stSelectbox > label, .stFileUploader label {
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--macaroon-subheader);
        margin-bottom: 0.5rem;
    }

    /* Input Fields (Text, Number, etc.) */
    .stTextInput>div>div>input, .stSelectbox>div>div, .stFileUploader>div>div>button {
        border-radius: 8px;
        border: 1px solid var(--macaroon-border);
        padding: 0.7rem;
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
    }
    .stFileUploader>div>div>button {
        background-color: var(--macaroon-light-bg);
        color: var(--macaroon-text);
        border: 1px solid var(--macaroon-border);
        box-shadow: none;
        transition: all 0.2s ease;
    }
    .stFileUploader>div>div>button:hover {
        background-color: #F0EDED; /* Slightly darker light bg */
        transform: none;
        box_shadow: none;
    }

    /* Alerts (Info, Success, Warning) */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
    }
    .stAlert.info { background-color: var(--macaroon-info-light); border-left: 5px solid var(--macaroon-info-dark); color: var(--macaroon-text); }
    .stAlert.success { background-color: var(--macaroon-success-light); border-left: 5px solid var(--macaroon-success-dark); color: var(--macaroon-text); }
    .stAlert.warning { background-color: var(--macaroon-warning-light); border-left: 5px solid var(--macaroon-warning-dark); color: var(--macaroon-text); }
    .stAlert.error { background-color: var(--macaroon-error-light); border-left: 5px solid var(--macaroon-error-dark); color: var(--macaroon-text); }

    /* Custom Boxes for Results */
    .acne-result-box {
        background-color: #f7e6f0; /* Light pink for result box */
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #D8A7B1; /* Rose Pink border */
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .acne-recommendation-box {
        background-color: #e6f0f5; /* Light blueish for recommendation */
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #A8DADC; /* Soft Teal border */
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 1.5rem;
    }
    .journal-entry-box {
        background-color: #fff9e6; /* Creamy yellow for journal entries */
        border: 1px solid #ffecb3;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .journal-entry-box h4 {
        color: var(--macaroon-subheader);
        margin-bottom: 5px;
    }


    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: var(--macaroon-sidebar-bg); /* Light Grayish Pink */
        padding: 1.5rem;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    [data-testid="stSidebarContent"] .stMarkdown h2 {
        color: var(--macaroon-main-header);
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    [data-testid="stSidebarContent"] .stMarkdown p {
        font-size: 0.95rem;
        line-height: 1.5;
        color: var(--macaroon-text);
    }
    [data-testid="stSidebarContent"] .stSpinnerContainer {
        color: var(--macaroon-accent-pink); /* Spinner color */
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F0D5E6; /* Lighter pink for expander header */
        color: var(--macaroon-main-header);
        font-weight: 600;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
    }
    .streamlit-expanderContent {
        background-color: var(--macaroon-light-bg);
        border: 1px solid #F0D5E6;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }

    /* Footer */
    .footer {
        font-family: 'Poppins', sans-serif;
        font-size: 0.85rem;
        color: var(--macaroon-text);
        text-align: center;
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--macaroon-border);
    }

    /* Image Container (for centered image) */
    .stImage {
        display: flex;
        justify-content: center;
        margin-bottom: 1.5rem;
    }
    .stImage img {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar Content ---
with st.sidebar:
    st.markdown("## Petunjuk Penggunaan 📝")
    st.markdown("""
    1.  **Ambil/Unggah Gambar:** Gunakan kamera atau unggah gambar area kulit yang berjerawat di bagian tengah aplikasi.
    2.  **Pilih Tipe Kulit:** Berikan informasi tipe kulit Anda (opsional, tapi disarankan untuk rekomendasi lebih akurat).
    3.  **Mulai Deteksi:** Klik tombol "Mulai Deteksi Jerawat" untuk mendapatkan analisis dan rekomendasi personal.
    """)
    st.markdown("---") # Separator line

    st.markdown("## Tentang Aplikasi ✨")
    st.markdown("""
        **AcneSolve** memanfaatkan teknologi *deep learning* untuk menganalisis gambar jerawat Anda
        dan mengklasifikasikannya ke dalam beberapa jenis umum. Berdasarkan deteksi tersebut,
        aplikasi ini akan memberikan rekomendasi produk dan kandungan skincare yang sesuai.
        
        **Catatan Penting:**
        * Hasil deteksi adalah perkiraan dan bukan pengganti diagnosis medis.
        * Selalu konsultasikan dengan dokter kulit atau dermatologis untuk masalah kulit yang serius atau persisten.
        
        Aplikasi ini dibuat dengan ❤️ dan Streamlit.
    """)
    st.markdown("---") # Separator line

    with st.expander("💡 FAQ & Tips Cepat Seputar Jerawat"):
        st.markdown("""
        **Apa itu jerawat?**
        Jerawat adalah kondisi kulit yang terjadi ketika folikel rambut tersumbat oleh minyak dan sel kulit mati, menyebabkan komedo, jerawat, atau kista.
        
        **Bisakah makanan memicu jerawat?**
        Beberapa penelitian menunjukkan diet tinggi gula dan produk susu bisa memperburuk jerawat pada beberapa orang. Namun, respons setiap individu berbeda.
        
        **Kapan harus ke dokter kulit?**
        Jika jerawat Anda parah, nyeri, tidak membaik dengan perawatan *over-the-counter*, atau meninggalkan bekas luka, segera konsultasi dengan dokter kulit.
        """)
    st.markdown("---") # Separator line

    st.markdown("## Mitra Jerawat 🧐 (Mitos vs. Fakta)")
    for i, pair in enumerate(MYTH_FACT_PAIRS):
        with st.expander(f"Mitos: {pair['myth']}"):
            st.markdown(f"**Fakta:** {pair['fact']}")
    st.markdown("---")

    st.markdown("## Cari Rekomendasi Lain 🔎")
    st.markdown("Find recommendations based on your criteria:")

    if df_pengobatan is not None:
        all_tingkat = df_pengobatan['tingkat'].unique().tolist()
        all_jenis_kulit = df_pengobatan['jenis kulit'].unique().tolist()
        all_kandungan_list = df_pengobatan['kandungan'].dropna().unique().tolist()
        all_kandungan_list = [k.strip() for k in all_kandungan_list if k.strip()]

        selected_tingkat_search = st.multiselect("Pilih Jenis Jerawat:", options=all_tingkat, default=[], key="search_tingkat")
        selected_jenis_kulit_search = st.multiselect("Pilih Tipe Kulit:", options=all_jenis_kulit, default=[], key="search_jenis_kulit")
        search_kandungan_text = st.text_input("Cari Kandungan (contoh: salicylic acid):", key="search_kandungan_input")

        if st.button("Cari Rekomendasi", key="perform_search_button"):
            if not selected_tingkat_search and not selected_jenis_kulit_search and not search_kandungan_text:
                st.warning("Select at least one criterion for search.")
            else:
                filtered_recom = df_pengobatan.copy()
                if selected_tingkat_search:
                    filtered_recom = filtered_recom[filtered_recom['tingkat'].isin(selected_tingkat_search)]
                if selected_jenis_kulit_search:
                    filtered_recom = filtered_recom[filtered_recom['jenis kulit'].isin(selected_jenis_kulit_search)]
                if search_kandungan_text:
                    filtered_recom = filtered_recom[filtered_recom['kandungan'].astype(str).str.contains(search_kandungan_text, case=False, na=False)]
                if not filtered_recom.empty:
                    st.success(f"Found {len(filtered_recom)} recommendations:")
                    for idx, row in filtered_recom.iterrows():
                        st.markdown(f"""
                        <div style="background-color: var(--macaroon-recommendation-box); padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid var(--macaroon-accent-teal);">
                            <p style="font-size:1rem; font-weight:bold; color:var(--macaroon-main-header);">Jenis Jerawat: {row['tingkat'].upper()}</p>
                            <p style="font-size:0.95rem; color:var(--macaroon-text);">Tipe Kulit: {row['jenis kulit'].capitalize()}</p>
                            <p style="font-size:0.95rem; color:var(--macaroon-text);">Kandungan Utama: {row['kandungan']}</p>
                            <p style="font-size:0.95rem; color:var(--macaroon-text);">Rekomendasi: {row['rekomendasi']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No recommendations matching your search criteria.")
    else:
        st.info("Recommendation data not loaded. Cannot perform search.")

header_container = st.container()
with header_container:
    col_logo, col_title = st.columns([0.1, 6])
    with col_logo:
        logo_path = "acenesence_logo.png"
        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            st.image(logo, width=800)
        else:
            st.warning(f"Logo '{logo_path}' not found.")
        
    #with col_title:
     #   st.markdown("<h1 class='main-header'>AcneSolve 🌸</h1>", unsafe_allow_html=True)
      #  st.write("""Detect your acne type and get personalized treatment recommendations for healthy skin!""")
    st.markdown("---")

st.markdown("<h2 class='subheader'>📸 Take or Upload Your Acne Image</h2>", unsafe_allow_html=True)

method = st.radio("How do you want to upload the image?", ("Gunakan Kamera", "Unggah Gambar dari Perangkat"), key="upload_method", horizontal=True)

img_data = None

if method == "Gunakan Kamera":
    st.info("💡 **Tips:** Ensure optimal lighting and focus the camera on the acne area.")
    img_file_buffer = st.camera_input("Klik untuk mengambil gambar")
    if img_file_buffer is not None:
        img_data = img_file_buffer.read()
        st.image(img_data, caption="Image from Camera.", use_column_width=True)
        st.success("✅ Image captured successfully!")
    else:
        st.warning("Please capture an image to start detection.")
elif method == "Unggah Gambar dari Perangkat":
    st.info("⬆️ **Tips:** Unggah gambar jelas berformat JPG, JPEG, atau PNG.")
    uploaded_file = st.file_uploader("Select your acne image", type=["jpg", "jpeg", "png"], key="image_uploader")
    if uploaded_file is not None:
        img_data = uploaded_file.read()
        st.image(img_data, caption="Uploaded image.", use_column_width=True)
        st.success("✅ Image uploaded successfully!")
    else:
        st.warning("Please upload an image to start detection.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h2 class='subheader'>🧴 Your Skin Type Information (Optional)</h2>", unsafe_allow_html=True)

st.write("""Select your skin type to help us provide more accurate recommendations.""")

skin_type = st.selectbox("What is your facial skin type?", ("Tidak Tahu / Lewati", "Normal", "Berminyak", "Kering", "Kombinasi", "Sensitive"), key="skin_type_selector")

if skin_type != "Tidak Tahu / Lewati":
    st.info(f"You selected skin type: **{skin_type}**")

st.markdown("---")

col_button_detect, col_button_clear = st.columns(2)

with col_button_detect:
    if st.button("Start Acne Detection", key="detect_button", use_container_width=True):
        if model is None or class_names is None or df_pengobatan is None:
            st.error("❌ App not ready: Model or recommendation data failed to load. Check sidebar for errors.")
        elif img_data is None:
            st.warning("Please upload or capture an image first.")
        else:
            with st.spinner("⏳ Analyzing your image. Please wait..."):
                try:
                    image = Image.open(io.BytesIO(img_data)).convert("RGB")
                    size = (150, 150) # Matching model input_shape
                    image = ImageOps.fit(image, size, Image.LANCZOS)
                    image_array = np.asarray(image)
                    
                    preprocessed_image_array = tf.keras.applications.efficientnet.preprocess_input(image_array.astype(np.float32))

                    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
                    data[0] = preprocessed_image_array
                    
                    acne_prediction = model.predict(data)
                    
                    st.session_state['acne_prediction_results'] = {
                        'prediction': acne_prediction,
                        'img_data': img_data, # Simpan data gambar asli di session_state
                        'skin_type': skin_type
                    }
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during image processing or prediction: {e}")
                    st.info("Ensure the image is clear and suitable for detection. Also check the console for error details.")

with col_button_clear:
    if st.button("Restart / Clear", key="clear_button", use_container_width=True):
        if 'acne_prediction_results' in st.session_state:
            del st.session_state['acne_prediction_results']
        st.experimental_rerun()

if 'acne_prediction_results' in st.session_state and st.session_state['acne_prediction_results'] is not None:
    st.markdown("<h2 class='subheader'>✨ Your Skin Analysis Results</h2>", unsafe_allow_html=True)
    
    results = st.session_state['acne_prediction_results']
    acne_prediction = results['prediction']
    skin_type = results['skin_type']
    original_img_data = results['img_data'] # Ambil data gambar dari session_state

    acne_index = np.argmax(acne_prediction)
    acne_class_raw = class_names[acne_index].strip()
    acne_confidence = acne_prediction[0][acne_index]

    acne_class_for_csv = acne_class_raw.lower()
    if acne_class_for_csv == 'blackheads': acne_class_for_csv = 'blackhead'
    elif acne_class_for_csv == 'whiteheads': acne_class_for_csv = 'whitehead'
    elif acne_class_for_csv == 'papules': acne_class_for_csv = 'papule'
    elif acne_class_for_csv == 'pustules': acne_class_for_csv = 'pustule'
    elif acne_class_for_csv == 'cystic': acne_class_for_csv = 'cystic'
    elif acne_class_for_csv == 'lv0': acne_class_for_csv = 'lv0'

    st.markdown(f"""
    <div class="acne-result-box">
        <p style="font-size:1.2rem; font-weight:bold; color:var(--macaroon-main-header);">Main Detection: <b>{acne_class_raw.upper()}</b></p>
        <p style="font-size:1rem; color:var(--macaroon-main-header);">Confidence Score: {acne_confidence:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    if acne_class_raw.lower() == 'lv0':
        st.markdown("""
        <div style="background-color: var(--macaroon-success-light); padding: 15px; border-radius: 8px; border-left: 5px solid var(--macaroon-success-dark); margin-top:1rem;">
            <p style="font-size:1.1rem; color:var(--macaroon-text); font-weight:bold;">✨ Congratulations! Your skin appears clear of significant acne.</p>
            <p style="font-size:0.95rem; color:var(--macaroon-text);">Maintain your skin's cleanliness and health by regularly cleansing and moisturizing according to your skin type.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p>You are detected with acne type: <b>{acne_class_raw.upper()}</b>. Here's a brief description:</p>
        <ul>
            <li><b>Blackhead:</b> An open clogged pore, exposed to air, turning black.</li>
            <li><b>Pustules:</b> Pus-filled pimples, red bumps with a white/yellow center.</li>
            <li><b>Whitehead:</b> Closed clogged pores, not exposed to air, remaining white.</li>
            <li><b>Papules:</b> Small, red, tender bumps, without pus.</li>
            <li><b>Cystic:</b> Severe acne, large, red, painful, pus-filled lumps deep under the skin. High risk of scarring.</li>
        </ul>
        """, unsafe_allow_html=True)

    st.markdown("### Detail Prediction Score:")
    fig, ax = plt.subplots(figsize=(8, 4))
    classes_display = [c.replace('lv0', 'No Acne').upper() for c in class_names]
    confidences = acne_prediction[0]
    
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_classes = [classes_display[i] for i in sorted_indices]
    sorted_confidences = [confidences[i] for i in sorted_indices]

    ax.barh(sorted_classes, sorted_confidences, color='#A8DADC')
    ax.set_xlabel("Confidence Score")
    ax.set_title("Probability Prediction of Acne Types")
    ax.set_xlim(0, 1)
    for i, v in enumerate(sorted_confidences):
        ax.text(v + 0.02, i, f"{v:.2f}", color=plt.rcParams['text.color'], va='center')
    
    ax.tick_params(axis='x', colors=plt.rcParams['text.color'])
    ax.tick_params(axis='y', colors=plt.rcParams['text.color'])
    ax.xaxis.label.set_color(plt.rcParams['text.color'])
    ax.yaxis.label.set_color(plt.rcParams['text.color'])
    ax.title.set_color(plt.rcParams['text.color'])
    
    fig.patch.set_facecolor(st.get_option("theme.backgroundColor"))
    ax.set_facecolor(st.get_option("theme.backgroundColor"))

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 🌿 Skincare Recommendations for Your Skin:")
    
    if df_pengobatan is not None:
        user_skin_type_csv = SKIN_TYPE_MAP_TO_CSV.get(skin_type, "semua") 

        filtered_by_acne = df_pengobatan[df_pengobatan['tingkat'] == acne_class_for_csv]

        recom_row = None
        
        specific_skin_recom = filtered_by_acne[filtered_by_acne['jenis kulit'] == user_skin_type_csv]
        if not specific_skin_recom.empty:
            recom_row = specific_skin_recom.iloc[0]
        else:
            general_skin_recom = filtered_by_acne[filtered_by_acne['jenis kulit'] == 'semua']
            if not general_skin_recom.empty:
                recom_row = general_skin_recom.iloc[0]
        
        if recom_row is not None:
            st.markdown(f"""
            <div class="acne-recommendation-box">
                <p style="font-size:1.15rem; font-weight:bold; color:var(--macaroon-subheader);">Skincare Recommendation for <b>{acne_class_raw.upper()}</b> ({skin_type} Skin):</p>
            """, unsafe_allow_html=True)
            
            kandungan_wajib = recom_row['kandungan']
            rekomendasi_tambahan = recom_row['rekomendasi']

            if pd.notna(kandungan_wajib):
                st.markdown(f"<p style='font-size:1rem;'><b>Key Ingredients:</b> {kandungan_wajib}</p>", unsafe_allow_html=True)
            if pd.notna(rekomendasi_tambahan):
                st.markdown(f"<p style='font-size:1rem;'><b>Additional Tips:</b> {rekomendasi_tambahan}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.warning(f"Sorry, specific recommendations for acne type **{acne_class_raw.upper()}** and skin type **{skin_type}** are not yet available in our database.")
            st.info("Try selecting 'Not Sure / Skip' for Skin Type for general recommendations if available.")
    else:
        st.error("Cannot provide recommendations as treatment data is not loaded.")

    st.warning("⚠️ **Important:** These recommendations are general and not a substitute for professional advice. Always **consult a dermatologist** for the most suitable diagnosis and treatment plan for your skin condition.")

    st.markdown("---")
    st.markdown("### Enhance Your Understanding of Acne 📚")
    with st.expander(f"Learn More about {acne_class_raw.upper()} & Its Treatment"):
        if acne_class_raw.lower() == 'blackhead':
            st.markdown("""
            **Blackhead:** An open clogged pore, exposed to the skin's surface, oxidizing and turning black.
            **Causes:** Excess sebum production, dead skin cell buildup, bacteria.
            **Treatment Tips:**
            * Use a cleanser with Salicylic Acid (BHA) to dissolve pore blockages.
            * Topical retinoids can help accelerate skin cell turnover.
            * Avoid squeezing blackheads, as it can worsen inflammation.
            """)
        elif acne_class_raw.lower() == 'whitehead':
            st.markdown("""
            **Whitehead:** Similar to a blackhead, but this clogged pore is closed by a layer of skin, so it's not exposed to air and remains white.
            **Causes:** Same as blackheads, sebum and dead skin cells trapped.
            **Treatment Tips:**
            * Gentle exfoliation with AHA (Glycolic Acid, Lactic Acid) to remove dead skin cells.
            * Benzoyl Peroxide or Salicylic Acid can help.
            * Do not try to forcefully extract whiteheads.
            """)
        elif acne_class_raw.lower() == 'papule':
            st.markdown("""
            **Papules:** Small, red, tender bumps with no pus at their tips. This is an early stage of acne inflammation.
            **Causes:** Clogged pores inflamed by bacteria.
            **Treatment Tips:**
            * Benzoyl Peroxide is effective for killing acne-causing bacteria.
            * Topical antibiotics (by prescription) may be needed to reduce inflammation.
            * Avoid rubbing or irritating the area.
            """)
        elif acne_class_raw.lower() == 'pustule':
            st.markdown("""
            **Pustules:** Pus-filled pimples, red bumps with a white or yellow center.
            **Causes:** Further inflammation of papules, where bacteria cause pus to form.
            **Treatment Tips:**
            * Use Benzoyl Peroxide or Salicylic Acid.
            * Oral or topical antibiotics (by prescription) may be necessary for severe cases.
            * DO NOT SQUEEZE THEM! This can spread bacteria and cause scarring.
            """)
        elif acne_class_raw.lower() == 'cystic':
            st.markdown("""
            **Cystic Acne:** The most severe form of acne, appearing as large, red, very painful, pus-filled lumps deep under the skin. High risk of leaving scars.
            **Causes:** Deep inflammation involving the rupture of hair follicles under the skin.
            **Treatment Tips:**
            * **Must consult a dermatologist!** Treatment usually involves oral antibiotics, isotretinoin, or corticosteroid injections.
            * Never attempt to squeeze cystic acne.
            """)
        elif acne_class_raw.lower() == 'lv0':
            st.markdown("""
            **No acne detected / Level 0 (Lv0):** This means your skin is in good condition with no significant active acne indications.
            **Treatment Tips:**
            * Continue your basic skincare routine: cleansing, moisturizing, and using sunscreen.
            * Choose products suitable for your skin type.
            * Jaga pola makan sehat dan hidrasi yang cukup untuk mendukung kesehatan kulit.
            """)
        else:
            st.markdown("More information about this acne type will be added soon!")

st.markdown("---")

st.markdown(f"""
<div style="background-color: var(--macaroon-info-light); padding: 15px; border-radius: 8px; border-left: 5px solid var(--macaroon-info-dark); margin-top:2rem;">
    <p style="font-size:1.1rem; color:var(--macaroon-text); font-weight:bold;">✨ Did You Know?</p>
    <p style="font-size:1rem; color:var(--macaroon-text);">{random.choice(DID_YOU_KNOW_FACTS)}</p>
</div>
""", unsafe_allow_html=True)


st.markdown("---")
st.markdown("<h2 class='subheader'>🗓️ Your Skin Progress Journal</h2>", unsafe_allow_html=True)
st.markdown("""Track your skin changes over time! Upload photos regularly and add brief notes.""")

uploaded_journal_file = st.file_uploader("Upload progress photo (Optional)", type=["jpg", "jpeg", "png"], key="journal_uploader")
journal_note = st.text_area("Note for this progress:", key="journal_note_input")

if st.button("Save Progress", key="save_journal_button"):
    if uploaded_journal_file is not None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = uploaded_journal_file.name.split('.')[-1]
            image_filename = f"progress_image_{timestamp}.{file_extension}"
            image_path = os.path.join(JOURNAL_DIR, image_filename)
            
            with open(image_path, "wb") as f:
                f.write(uploaded_journal_file.getbuffer())
            
            new_entry = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": image_filename,
                "note": journal_note if journal_note else "No note."
            }
            journal_entries.append(new_entry)
            save_journal_entries(journal_entries)
            st.success("✅ Progress saved successfully!")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"❌ Failed to save progress: {e}")
    else:
        st.warning("Please upload a photo to save journal progress.")

# --- Bagian untuk menyimpan foto deteksi ke jurnal ---
if 'acne_prediction_results' in st.session_state and st.session_state['acne_prediction_results'] is not None:
    original_img_data_for_journal = st.session_state['acne_prediction_results']['img_data']
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Simpan Foto Deteksi Ini ke Jurnal Anda</h3>", unsafe_allow_html=True)
    journal_note_from_detection = st.text_input("Tambahkan catatan untuk foto hasil deteksi ini (Opsional):", key="journal_note_from_detection")
    
    if st.button("Simpan Foto Deteksi Ini ke Jurnal", key="save_detection_to_journal_button"):
        if original_img_data_for_journal is not None:
            success, message = add_journal_entry(original_img_data_for_journal, journal_note_from_detection)
            if success:
                st.success(f"✅ {message}")
                st.experimental_rerun() # Refresh halaman untuk menampilkan entri baru
            else:
                st.error(f"❌ {message}")
        else:
            st.warning("Tidak ada gambar deteksi yang tersedia untuk disimpan.")


st.markdown("---")
st.markdown("### Riwayat Progres Kulit:")

if journal_entries:
    sorted_entries = sorted(journal_entries, key=lambda x: x['date'], reverse=True)
    
    for entry in sorted_entries:
        date_obj = datetime.strptime(entry['date'], "%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
        <div class="journal-entry-box">
            <h4>📅 {date_obj.strftime('%d %B %Y - %H:%M')}</h4>
        """, unsafe_allow_html=True)
        
        full_image_path = os.path.join(JOURNAL_DIR, entry['image_path'])
        if os.path.exists(full_image_path):
            st.image(full_image_path, caption=f"Progress Photo Date {date_obj.strftime('%d/%m/%Y')}", use_column_width=True)
        else:
            st.warning(f"Image not found: {entry['image_path']}")
        
        st.markdown(f"<p><b>Note:</b> {entry['note']}</p>", unsafe_allow_html=True)
        
        if st.button(f"Delete This Entry ({date_obj.strftime('%H:%M:%S')})", key=f"delete_journal_entry_{entry['date']}"):
            journal_entries.remove(entry)
            save_journal_entries(journal_entries)
            if os.path.exists(full_image_path):
                os.remove(full_image_path)
                st.info(f"Image '{entry['image_path']}' deleted successfully.")
            st.success("Progress entry deleted successfully.")
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No skin progress entries yet. Start adding your progress!")

st.markdown("""
<div class='footer'>
    <p>AcneSolve v1.0 &copy; 2025. Made with ❤️ and Streamlit.</p>
</div>
""", unsafe_allow_html=True)