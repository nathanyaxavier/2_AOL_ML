import streamlit as st
import pandas as pd
import joblib
from PIL import Image

@st.cache_resource
def load_model():
    model = joblib.load("best_random_forest_model.pkl")
    scaler = joblib.load("robust_standard_scaler.pkl")
    columns = joblib.load("feature_columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_model()

st.set_page_config(
    page_title="Paddy Yield Prediction",
    layout="wide"
)

st.markdown("""
    <style>
    /* ==========================================================================
       1. PENGATURAN GLOBAL & ADAPTIF TEMA (LIGHT/DARK MODE)
       ========================================================================== */
    
    /* Default / Gaya untuk Light Mode */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background-color: #FDFBF7;
    }
    .input-card {
        background-color: #FFEBBC;
        padding: 24px;
        border-radius: 12px;
        border-left: 6px solid #B3BC53;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .card-title {
        color: #768C3A;
        font-size: 24px;
        font-weight: 700;
        margin-top: 0px;
        margin-bottom: 5px;
    }
    .card-caption {
        font-size: 14px; 
        color: #555555; 
        margin-bottom: 15px;
    }
    div[data-testid="stNumberInput"] input {
        background-color: #FFF2D4 !important;
        color: #333333 !important;
        border: 1px solid #FFDA8C !important;
        border-radius: 6px !important;
        font-weight: 500;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #FFDA8C !important;
        color: #768C3A !important;
        border: 1px solid #FFDA8C !important;
    }
    .result-card {
        background-color: #FFF2D4;
        border: 2px solid #F7A503;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 4px 15px rgba(247, 165, 3, 0.15);
    }
    .recommendation-box {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-left: 6px solid #768C3A;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        color: #333333;
    }

    /* Penyesuaian Otomatis Ketika User Menggunakan Dark Mode */
    @media (prefers-color-scheme: dark) {
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #121212;
        }
        .input-card {
            background-color: #2D271E; /* Versi gelap dari Vanilla Cream */
            border-left: 6px solid #768C3A;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .card-title {
            color: #FFDA8C; /* Sunny Wheat agar kontras di latar gelap */
        }
        .card-caption {
            color: #CCCCCC;
        }
        div[data-testid="stNumberInput"] input {
            background-color: #1E1A13 !important;
            color: #FFFFFF !important;
            border: 1px solid #768C3A !important;
        }
        div[data-testid="stNumberInput"] button {
            background-color: #3A3225 !important;
            color: #FFDA8C !important;
            border: 1px solid #768C3A !important;
        }
        .result-card {
            background-color: #2D210A; /* Versi gelap dari Amber/Yellow */
            border: 2px solid #F7A503;
            box-shadow: 0 4px 15px rgba(247, 165, 3, 0.3);
        }
        .recommendation-box {
            background-color: #1E1E1E;
            border: 1px solid #333333;
            border-left: 6px solid #B3BC53;
            color: #E0E0E0;
        }
    }

    /* ==========================================================================
       2. PENGATURAN ELEMEN STATIS (BERLAKU DI KEDUA MODE)
       ========================================================================== */
    h1, h2, h3 {
        color: #768C3A !important;
        font-weight: 700;
    }
    div.stButton > button:first-child {
        background-color: #768C3A;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div.stButton > button:first-child:hover {
        background-color: #B3BC53;
        color: white;
        border: none;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .result-value {
        color: #768C3A;
        font-size: 36px;
        font-weight: 800;
        margin-top: 10px;
    }
    .status-card {
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
        font-size: 18px;
    }
    .status-tinggi { background-color: #D4EDDA; color: #155724; border: 1px solid #C3E6CB; }
    .status-sedang { background-color: #FFF3CD; color: #856404; border: 1px solid #FFEEBA; }
    .status-rendah { background-color: #F8D7DA; color: #721C24; border: 1px solid #F5C6CB; }
    
    .recommendation-item {
        margin-bottom: 12px;
        line-height: 1.6;
    }
    .recommendation-title {
        font-weight: bold;
        color: #B3BC53;
    }
    </style>
""", unsafe_allow_html=True)

try:
    img = Image.open("logo_padi.jpg")
    width, height = img.size
    img_cropped = img.crop((0, 150, 735, 400))
    st.image(img_cropped, use_container_width=True)
except Exception:
    pass


st.title("Paddy Yield Prediction System")
st.write(
    """
    Sistem ini digunakan untuk memperkirakan hasil panen padi berdasarkan
    kondisi lahan, penggunaan benih, pemupukan, serta perlindungan tanaman.
    """
)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="input-card">
            <div class="card-title">Luas Lahan Utama (Hectares)</div>
            <div class="card-caption">
                Masukkan total luas lahan sawah aktif yang digunakan untuk budidaya padi pada musim tanam saat ini.
            </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Luas Lahan (ha)**")
    st.caption("Luas total area sawah aktif yang Anda tanami padi saat ini.")
    hectares = st.number_input(
        label="Luas Lahan (ha)",
        min_value=0.0,
        value=2.5,
        label_visibility="collapsed"
    )
    
    st.markdown("**Jumlah Benih Padi (kg)**")
    st.caption("Berat total benih yang disemai untuk kebutuhan musim tanam ini.")
    seedrate = st.number_input(
        label="Jumlah Benih Padi (kg)",
        min_value=0.0,
        value=150.0,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="input-card">
            <div class="card-title">Area Pembibitan Awal (Nursery)</div>
            <div class="card-caption">
                Masukkan informasi mengenai area pembibitan dan persiapan tanah yang digunakan sebelum dipindahkan ke lahan utama.
            </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Luas Area Pembibitan (Cents)**")
    st.caption("Luas petak tanah khusus yang Anda gunakan untuk menyemai benih awal sebelum dipindahkan ke sawah utama.")
    nursery_area = st.number_input(
        label="Luas Area Pembibitan (Cents)",
        min_value=0.0,
        value=120.0,
        label_visibility="collapsed"
    )
    
    st.markdown("**Persiapan Tanah Pembibitan (Ton)**")
    st.caption("Banyaknya pupuk organik atau kompos yang diberikan untuk mengelola tanah khusus di area pembibitan awal.")
    lp_nursery = st.number_input(
        label="Persiapan Tanah Pembibitan (Ton)",
        min_value=0.0,
        value=6.0,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="input-card">
            <div class="card-title">Pengolahan Lahan Utama</div>
            <div class="card-caption">
                Masukkan data terkait lahan sawah utama, termasuk penggunaan bahan organik dan pengelolaan sisa tanaman.
            </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Persiapan Lahan Utama (Ton)**")
    st.caption("Total berat pupuk organik, kompos, atau kapur dasar saat pertama kali membajak sawah utama.")
    lp_mainfield = st.number_input(
        label="Persiapan Lahan Utama (Ton)",
        min_value=0.0,
        value=75.0,
        label_visibility="collapsed"
    )
    
    st.markdown("**Pengelolaan Jerami / Sisa Sawah (Bundles)**")
    st.caption("Jumlah ikatan jerami atau sisa rumput kering hasil panen lalu yang dihamparkan kembali sebagai mulsa alami.")
    trash = st.number_input(
        label="Pengelolaan Jerami (Bundles)",
        min_value=0.0,
        value=540.0,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="input-card">
            <div class="card-title">Pemupukan & Nutrisi Tanaman</div>
            <div class="card-caption">
                Masukkan jumlah pupuk dan nutrisi yang diberikan pada berbagai fase pertumbuhan padi.
            </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Pupuk DAP Hari Ke-20 (Kg)**")
    st.caption("Dosis pupuk Di-ammonium Phosphate yang diberikan saat usia padi menginjak 20 hari setelah tanam.")
    dap = st.number_input(
        label="Pupuk DAP Hari Ke-20 (Kg)",
        min_value=0.0,
        value=240.0,
        label_visibility="collapsed"
    )
    
    st.markdown("**Pupuk Urea Hari Ke-40 (Kg)**")
    st.caption("Dosis pupuk Nitrogen (Urea) yang ditaburkan pada umur tanaman 40 hari guna memacu fase vegetatif.")
    urea = st.number_input(
        label="Pupuk Urea Hari Ke-40 (Kg)",
        min_value=0.0,
        value=162.78,
        label_visibility="collapsed"
    )
    
    st.markdown("**Pupuk Kalium / Potash Hari Ke-50 (Kg)**")
    st.caption("Dosis pupuk Kalium (Potash/KCL) yang ditaburkan pada umur 50 hari untuk membantu pengisian bulir.")
    potash = st.number_input(
        label="Pupuk Kalium / Potash Hari Ke-50 (Kg)",
        min_value=0.0,
        value=62.28,
        label_visibility="collapsed"
    )
    
    st.markdown("**Nutrisi Mikro Hari Ke-70 (Kg)**")
    st.caption("Dosis suplemen zat hara mikro (seng, besi, mangan) yang disemprotkan pada umur 70 hari.")
    micronutrients = st.number_input(
        label="Nutrisi Mikro Hari Ke-70 (Kg)",
        min_value=0.0,
        value=90.0,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="input-card">
            <div class="card-title">Perlindungan Tanaman</div>
            <div class="card-caption">
                Masukkan data penggunaan herbisida dan pestisida yang diterapkan selama masa tanam.
            </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Herbisida Hari Ke-28 (Litre)**")
    st.caption("Dosis cairan pembasmi rumput liar/gulma (berbahan kimia Thiobencarb) pada hari ke-28.")
    weed = st.number_input(
        label="Herbisida Hari Ke-28 (Litre)",
        min_value=0.0,
        value=12.0,
        label_visibility="collapsed"
    )
    
    st.markdown("**Pestisida Hari Ke-60 (ml)**")
    st.caption("Volume cairan pembasmi serangga atau hama penyakit yang disemprotkan pada umur 60 hari.")
    pest = st.number_input(
        label="Pestisida Hari Ke-60 (ml)",
        min_value=0.0,
        value=3600.0,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

predict_button = st.button("Hitung Prediksi Hasil Panen")

if predict_button:
    input_df = pd.DataFrame({
        'Hectares ': [hectares],
        'Micronutrients_70Days': [micronutrients],
        'Potassh_50Days': [potash],
        'Urea_40Days': [urea],
        'Pest_60Day(in ml)': [pest],
        'LP_Mainfield(in Tonnes)': [lp_mainfield],
        'DAP_20days': [dap],
        'Trash(in bundles)': [trash],
        'Seedrate(in Kg)': [seedrate],
        'LP_nurseryarea(in Tonnes)': [lp_nursery],
        'Weed28D_thiobencarb': [weed],
        'Nursery area (Cents)': [nursery_area]
    })

    try:
        input_df = input_df.reindex(columns=columns, fill_value=0)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        yield_per_hectare = prediction / hectares if hectares > 0 else 0
        
        if yield_per_hectare < 4000:
            status_class = "status-rendah"
            status_label = "Rendah"
        elif 4000 <= yield_per_hectare <= 6000:
            status_class = "status-sedang"
            status_label = "Sedang"
        else:
            status_class = "status-tinggi"
            status_label = "Tinggi"

        st.markdown(f"""
            <div class="result-card">
                <div style="font-size: 18px; color: #555555; text-transform: uppercase; letter-spacing: 1px; font-weight: bold;">
                    Estimasi Total Hasil Panen Padi
                </div>
                <div class="result-value">
                    {prediction:,.2f} Kg
                </div>
                <div style="margin-top: 15px; font-size: 15px; color: #666666;">
                    Rasio Produktivitas Lahan saat ini: <strong>{yield_per_hectare:,.2f} Kg/Hektar</strong>
                </div>
                <div style="display: flex; justify-content: center; align-items: center; margin-top: 10px;">
                    <div style="font-size: 15px; color: #666666; margin-right: 10px;">Tingkat Produktivitas:</div>
                    <div class="status-card {status_class}">{status_label}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.subheader("Analisis Hasil & Rekomendasi Budidaya")
        
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        
        rasio_benih = seedrate / hectares if hectares > 0 else 0
        st.markdown('<div class="recommendation-item">', unsafe_allow_html=True)
        if rasio_benih < 20:
            st.markdown("<span class='recommendation-title'>Manajemen Benih:</span> Penggunaan benih Anda tergolong rendah per hektarnya (kurang dari 20 kg/ha). Disarankan untuk mengoptimalkan populasi tanaman dengan menambah kapasitas benih berkualitas tinggi agar ruang lahan utama termanfaatkan dengan maksimal.", unsafe_allow_html=True)
        elif rasio_benih > 60:
            st.markdown("<span class='recommendation-title'>Manajemen Benih:</span> Jumlah benih per hektar terpantau sangat tinggi. Hal ini berisiko memicu kepadatan tanaman yang berlebihan, mempermudah penyebaran hama, serta meningkatkan kompetisi nutrisi antar batang padi.", unsafe_allow_html=True)
        else:
            st.markdown("<span class='recommendation-title'>Manajemen Benih:</span> Proporsi sebaran benih terhadap luas lahan utama sudah berada dalam kondisi ideal untuk mendukung pertumbuhan populasi tanaman.", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="recommendation-item">', unsafe_allow_html=True)
        if lp_mainfield / hectares < 20:
            st.markdown("<span class='recommendation-title'>Kesuburan Tanah Utama:</span> Alokasi pupuk organik/kompos dasar pada pengolahan lahan utama relatif minim. Pertimbangkan peningkatan volume pupuk organik di awal pembajakan berikutnya guna memperbaiki struktur pori tanah dan kapasitas tukar kation.", unsafe_allow_html=True)
        else:
            st.markdown("<span class='recommendation-title'>Kesuburan Tanah Utama:</span> Pengaplikasian bahan organik dasar pada pengolahan tanah utama terpantau optimal, yang secara nyata membantu stabilitas ekosistem mikroba tanah.", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="recommendation-item">', unsafe_allow_html=True)
        rekomendasi_nutrisi = []
        if urea < 150:
            rekomendasi_nutrisi.append("Menambah takaran pupuk Nitrogen (Urea) secara berkala pada fase vegetatif hari ke-40 jika daun tampak pucat.")
        if potash < 50:
            rekomendasi_nutrisi.append("Mengoptimalkan unsur Kalium (Potash) di umur 50 hari untuk meminimalkan kerontokan serta memperkuat dinding sel tanaman.")
        if dap < 200:
            rekomendasi_nutrisi.append("Memastikan kecukupan hara Fosfat (DAP) pada awal tanam (hari ke-20) guna mempercepat multiplikasi akar.")
            
        if rekomendasi_nutrisi:
            st.markdown("<span class='recommendation-title'>Strategi Pemupukan Makro:</span> Berdasarkan kalkulasi dosis, Anda disarankan untuk: " + ", ".join(rekomendasi_nutrisi) + ".", unsafe_allow_html=True)
        else:
            st.markdown("<span class='recommendation-title'>Strategi Pemupukan Makro:</span> Pemberian kombinasi pupuk Urea, DAP, dan Potash telah seimbang, memberikan pondasi nutrisi kuat pada fase kritis pertumbuhan.", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="recommendation-item">', unsafe_allow_html=True)
        if micronutrients > 70:
            st.markdown("<span class='recommendation-title'>Nutrisi Mikro:</span> Penyemprotan suplemen nutrisi mikro pada hari ke-70 terbukti membantu efisiensi metabolisme tanaman dalam pengisian kualitas bulir padi pada fase generatif.", unsafe_allow_html=True)
        else:
            st.markdown("<span class='recommendation-title'>Nutrisi Mikro:</span> Pemberian unsur mikro hara terpantau minimal. Untuk musim berikutnya, pertimbangkan penambahan seng (Zn) atau besi (Fe) pada fase pengisian bulir demi menghindari bulir padi hampa.", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="recommendation-item">', unsafe_allow_html=True)
        if weed < 10:
            st.markdown("<span class='recommendation-title'>Perlindungan Gulma:</span> Aplikasi herbisida pra/purna tumbuh di bawah ambang standar. Awasi pertumbuhan gulma liar di hari ke-28 agar tidak merebut jatah pupuk utama tanaman padi.", unsafe_allow_html=True)
        elif pest < 2500:
            st.markdown("<span class='recommendation-title'>Perlindungan Hama:</span> Intensitas perlindungan hama cair di bawah rata-rata nasional untuk skala lahan Anda. Lakukan monitoring ketat terhadap vektor penyakit seperti wereng atau penggerek batang menjelang hari ke-60.", unsafe_allow_html=True)
        else:
            st.markdown("<span class='recommendation-title'>Perlindungan Tanaman:</span> Langkah preventif perlindungan terhadap intervensi gulma pengganggu dan serangan hama penyakit sudah dikelola dengan intensif.", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan teknis saat melakukan kalkulasi prediksi: {e}")

st.markdown("---")

st.info(
    """
    Catatan Penggunaan:
    - Masukkan seluruh data numerik sesuai dengan kondisi aktual operasional pertanian Anda di lapangan.
    - Format penulisan angka desimal wajib menggunakan tanda titik (.) sebagai pemisah (contoh: 62.28).
    """
)