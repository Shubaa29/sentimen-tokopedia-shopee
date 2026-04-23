import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.graph_objects as go

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
# TAMPILAN APLIKASI
# =====================
st.set_page_config(
    page_title="Analisis Sentimen Tokopedia & Shopee",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Analisis Sentimen Ulasan Tokopedia & Shopee")
st.markdown("Aplikasi analisis sentimen ulasan pengguna menggunakan model **IndoBERT**")

# Sidebar informasi model
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
tab1, tab2 = st.tabs(["📝 Prediksi Teks", "📊 Prediksi Banyak Teks"])

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

            # Tampilkan hasil
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
                # Grafik confidence
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

            import pandas as pd
            df_result = pd.DataFrame(results)
            st.dataframe(df_result, use_container_width=True)

            # Pie chart distribusi
            counts = df_result['Sentimen'].value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index,
                values=counts.values,
                marker_colors=['green', 'red', 'gold']
            ))
            fig.update_layout(title="Distribusi Sentimen")
            st.plotly_chart(fig, use_container_width=True)
