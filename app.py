import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.graph_objects as go
import pandas as pd

# =====================
# KONFIGURASI
# =====================
MODEL_NAME = "George6767/indobert-sentimen-tokopedia-shopee"
LABEL_NAMES = ['negatif', 'netral', 'positif']
MAX_LEN = 128

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

# =====================
# FUNGSI PREDIKSI
# =====================
def prediksi(teks, tokenizer, model):
    encoding = tokenizer(
        teks,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
    return LABEL_NAMES[pred], probs.tolist()

# =====================
# DATA TEST CASE
# =====================
TEST_CASES = [
    # Positif
    {"id": "TC-POS-01", "kategori": "Positif", "input": "aplikasi tokopedia sangat mudah digunakan dan pengiriman cepat", "expected": "positif"},
    {"id": "TC-POS-02", "kategori": "Positif", "input": "fitur shopee pay sangat membantu belanja jadi lebih hemat", "expected": "positif"},
    {"id": "TC-POS-03", "kategori": "Positif", "input": "barang sampai cepat sesuai estimasi packing rapi dan aman", "expected": "positif"},
    {"id": "TC-POS-04", "kategori": "Positif", "input": "penjual sangat ramah dan responsif barang sesuai deskripsi", "expected": "positif"},
    {"id": "TC-POS-05", "kategori": "Positif", "input": "tampilan aplikasi tokopedia baru sangat bersih dan mudah dipahami", "expected": "positif"},
    {"id": "TC-POS-06", "kategori": "Positif", "input": "5 bintang aplikasi terbaik untuk belanja online di indonesia", "expected": "positif"},
    # Negatif
    {"id": "TC-NEG-01", "kategori": "Negatif", "input": "shopee sering error dan loading lama sangat mengecewakan sekali", "expected": "negatif"},
    {"id": "TC-NEG-02", "kategori": "Negatif", "input": "pembayaran gagal terus padahal saldo sudah cukup sangat menjengkelkan", "expected": "negatif"},
    {"id": "TC-NEG-03", "kategori": "Negatif", "input": "barang belum sampai sudah seminggu lebih tidak ada update tracking", "expected": "negatif"},
    {"id": "TC-NEG-04", "kategori": "Negatif", "input": "barang sampai dalam kondisi rusak packing tidak aman sama sekali", "expected": "negatif"},
    {"id": "TC-NEG-05", "kategori": "Negatif", "input": "customer service tidak responsif dan tidak membantu menyelesaikan masalah", "expected": "negatif"},
    {"id": "TC-NEG-06", "kategori": "Negatif", "input": "seller penipu barang tidak sesuai deskripsi foto beda jauh dengan aslinya", "expected": "negatif"},
    # Netral
    {"id": "TC-NET-01", "kategori": "Netral", "input": "aplikasi biasa saja tidak ada yang terlalu istimewa", "expected": "netral"},
    {"id": "TC-NET-02", "kategori": "Netral", "input": "aplikasi ini bisa digunakan untuk belanja dan membayar tagihan", "expected": "netral"},
    {"id": "TC-NET-03", "kategori": "Netral", "input": "sebaiknya tambahkan fitur live chat antara pembeli dan penjual", "expected": "netral"},
    {"id": "TC-NET-04", "kategori": "Netral", "input": "tokopedia dan shopee sama-sama memiliki kelebihan dan kekurangan masing-masing", "expected": "netral"},
    # Edge Cases
    {"id": "TC-EDGE-01", "kategori": "Edge Case", "input": "bagus", "expected": "positif"},
    {"id": "TC-EDGE-02", "kategori": "Edge Case", "input": "APLIKASI INI SANGAT MENGECEWAKAN DAN SERING ERROR", "expected": "negatif"},
    {"id": "TC-EDGE-03", "kategori": "Edge Case", "input": "aplikasinya good banget tapi sometimes suka error juga sih", "expected": "netral"},
    {"id": "TC-EDGE-04", "kategori": "Edge Case", "input": "bagus sekali aplikasinya sering error dan loading lama luar biasa", "expected": "negatif"},
]

# =====================
# TAMPILAN APLIKASI
# =====================
st.set_page_config(
    page_title="Analisis Sentimen Tokopedia & Shopee",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Analisis Sentimen Ulasan Tokopedia & Shopee")
st.markdown("Aplikasi analisis sentimen ulasan pengguna menggunakan model **IndoBERT**")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    st.write("**Model:** IndoBERT")
    st.write("**Dataset:** Google Play Store")
    st.write("**Aplikasi:** Tokopedia & Shopee")
    st.write("**Accuracy:** 86%")
    st.write("**F1-Score:** 0.84")
    st.markdown("---")
    st.write("**Kelas Sentimen:**")
    st.write("🟢 Positif")
    st.write("🔴 Negatif")
    st.write("🟡 Netral")

# Tab
tab1, tab2, tab3 = st.tabs(["📝 Prediksi Teks", "📊 Prediksi Banyak Teks", "🧪 Test Case"])

# =====================
# TAB 1 - PREDIKSI SATU TEKS
# =====================
with tab1:
    st.subheader("Prediksi Sentimen Satu Ulasan")
    teks_input = st.text_area("Masukkan ulasan:", placeholder="Contoh: aplikasi tokopedia sangat mudah digunakan", height=150)

    if st.button("🔍 Analisis Sentimen", key="btn1"):
        if teks_input.strip() == "":
            st.warning("Masukkan teks ulasan terlebih dahulu!")
        else:
            with st.spinner("Menganalisis sentimen..."):
                tokenizer, model = load_model()
                label, probs = prediksi(teks_input, tokenizer, model)

            col1, col2 = st.columns(2)
            with col1:
                if label == 'positif':
                    st.success(f"✅ Sentimen: **POSITIF**")
                elif label == 'negatif':
                    st.error(f"❌ Sentimen: **NEGATIF**")
                else:
                    st.warning(f"⚠️ Sentimen: **NETRAL**")
                st.write(f"**Confidence:** {max(probs)*100:.2f}%")

            with col2:
                fig = go.Figure(go.Bar(
                    x=[p*100 for p in probs],
                    y=LABEL_NAMES,
                    orientation='h',
                    marker_color=['red', 'gold', 'green']
                ))
                fig.update_layout(
                    title="Distribusi Probabilitas",
                    xaxis_title="Probabilitas (%)",
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)

# =====================
# TAB 2 - PREDIKSI BANYAK TEKS
# =====================
with tab2:
    st.subheader("Prediksi Sentimen Banyak Ulasan")
    teks_banyak = st.text_area(
        "Masukkan beberapa ulasan (satu ulasan per baris):",
        placeholder="Aplikasi sangat bagus\nSering error dan lambat\nBiasa saja",
        height=200
    )

    if st.button("🔍 Analisis Semua", key="btn2"):
        if teks_banyak.strip() == "":
            st.warning("Masukkan teks ulasan terlebih dahulu!")
        else:
            ulasan_list = [u.strip() for u in teks_banyak.split('\n') if u.strip()]
            with st.spinner(f"Menganalisis {len(ulasan_list)} ulasan..."):
                tokenizer, model = load_model()
                results = []
                for ulasan in ulasan_list:
                    label, probs = prediksi(ulasan, tokenizer, model)
                    results.append({
                        'Ulasan': ulasan,
                        'Sentimen': label.upper(),
                        'Confidence': f"{max(probs)*100:.2f}%"
                    })

            df_result = pd.DataFrame(results)
            st.dataframe(df_result, use_container_width=True)

            counts = df_result['Sentimen'].value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index,
                values=counts.values,
                marker_colors=['green', 'red', 'gold']
            ))
            fig.update_layout(title="Distribusi Sentimen")
            st.plotly_chart(fig, use_container_width=True)

# =====================
# TAB 3 - TEST CASE
# =====================
with tab3:
    st.subheader("🧪 Pengujian Test Case Model")
    st.markdown("Halaman ini menguji performa model secara otomatis menggunakan data test case yang sudah ditentukan.")

    # Filter kategori
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        kategori_filter = st.multiselect(
            "Filter Kategori:",
            options=["Positif", "Negatif", "Netral", "Edge Case"],
            default=["Positif", "Negatif", "Netral", "Edge Case"]
        )
    with col_f2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_test = st.button("▶️ Jalankan Test Case", key="btn_test", use_container_width=True)

    st.markdown("---")

    # Jalankan otomatis
    if run_test:
        tokenizer, model = load_model()
        filtered_tc = [tc for tc in TEST_CASES if tc["kategori"] in kategori_filter]

        results = []
        progress = st.progress(0)
        status_text = st.empty()

        for i, tc in enumerate(filtered_tc):
            status_text.text(f"Menguji {tc['id']}... ({i+1}/{len(filtered_tc)})")
            label, probs = prediksi(tc["input"], tokenizer, model)
            confidence = max(probs) * 100
            status = "✅ PASS" if label == tc["expected"] else "❌ FAIL"
            results.append({
                "ID": tc["id"],
                "Kategori": tc["kategori"],
                "Input Ulasan": tc["input"],
                "Expected": tc["expected"].upper(),
                "Prediksi": label.upper(),
                "Confidence": f"{confidence:.2f}%",
                "Status": status
            })
            progress.progress((i + 1) / len(filtered_tc))

        status_text.text("Pengujian selesai!")
        progress.empty()

        df_test = pd.DataFrame(results)

        # Metrik ringkasan
        total = len(df_test)
        passed = len(df_test[df_test["Status"] == "✅ PASS"])
        failed = total - passed
        pass_rate = passed / total * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Test Case", total)
        m2.metric("✅ PASS", passed)
        m3.metric("❌ FAIL", failed)
        m4.metric("Pass Rate", f"{pass_rate:.1f}%")

        st.markdown("---")

        # Tabel hasil
        st.subheader("📋 Hasil Test Case")

        def highlight_status(val):
            if "PASS" in str(val):
                return "background-color: #f0fdf4; color: #16a34a; font-weight: bold"
            elif "FAIL" in str(val):
                return "background-color: #fef2f2; color: #dc2626; font-weight: bold"
            return ""

        def highlight_sentimen(val):
            if str(val) == "POSITIF":
                return "color: #16a34a; font-weight: bold"
            elif str(val) == "NEGATIF":
                return "color: #dc2626; font-weight: bold"
            elif str(val) == "NETRAL":
                return "color: #d97706; font-weight: bold"
            return ""

        styled_df = df_test.style\
            .applymap(highlight_status, subset=["Status"])\
            .applymap(highlight_sentimen, subset=["Expected", "Prediksi"])

        st.dataframe(styled_df, use_container_width=True, height=500)

        # Grafik per kategori
        st.markdown("---")
        st.subheader("📊 Hasil per Kategori")
        cat_results = df_test.groupby("Kategori")["Status"].apply(
            lambda x: (x == "✅ PASS").sum()
        ).reset_index()
        cat_results.columns = ["Kategori", "PASS"]
        cat_results["Total"] = df_test.groupby("Kategori").size().values
        cat_results["Pass Rate"] = (cat_results["PASS"] / cat_results["Total"] * 100).round(1)

        fig_cat = go.Figure(go.Bar(
            x=cat_results["Kategori"],
            y=cat_results["Pass Rate"],
            marker_color=["#16a34a" if r == 100 else "#d97706" for r in cat_results["Pass Rate"]],
            text=[f"{r}%" for r in cat_results["Pass Rate"]],
            textposition="outside"
        ))
        fig_cat.update_layout(
            title="Pass Rate per Kategori (%)",
            yaxis=dict(range=[0, 115]),
            height=350
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # =====================
    # FORM MANUAL
    # =====================
    st.markdown("---")
    st.subheader("✏️ Uji Manual Test Case")
    st.markdown("Masukkan teks dan label yang diharapkan untuk menguji model secara manual.")

    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        manual_input = st.text_area("Input Ulasan:", placeholder="Masukkan teks ulasan yang ingin diuji", height=100, key="manual_input")
    with col_m2:
        expected_label = st.selectbox("Label yang Diharapkan:", ["positif", "negatif", "netral"], key="expected_label")
        st.markdown("<br>", unsafe_allow_html=True)
        run_manual = st.button("🔍 Uji Sekarang", key="btn_manual", use_container_width=True)

    if run_manual:
        if manual_input.strip() == "":
            st.warning("Masukkan teks ulasan terlebih dahulu!")
        else:
            with st.spinner("Menganalisis..."):
                tokenizer, model = load_model()
                label, probs = prediksi(manual_input, tokenizer, model)

            confidence = max(probs) * 100
            is_pass = label == expected_label

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if is_pass:
                    st.success(f"✅ **PASS** — Prediksi sesuai ekspektasi")
                else:
                    st.error(f"❌ **FAIL** — Prediksi tidak sesuai ekspektasi")

                st.write(f"**Input:** {manual_input}")
                st.write(f"**Expected:** {expected_label.upper()}")
                st.write(f"**Prediksi:** {label.upper()}")
                st.write(f"**Confidence:** {confidence:.2f}%")

            with col_r2:
                fig_m = go.Figure(go.Bar(
                    x=[p*100 for p in probs],
                    y=LABEL_NAMES,
                    orientation='h',
                    marker_color=['red', 'gold', 'green']
                ))
                fig_m.update_layout(
                    title="Distribusi Probabilitas",
                    xaxis_title="Probabilitas (%)",
                    height=250
                )
                st.plotly_chart(fig_m, use_container_width=True)
