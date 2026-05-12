import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
from io import StringIO

# ── Gensim ──────────────────────────────────────────────────────────────────
from gensim import corpora
import gensim

# ── Sastrawi ─────────────────────────────────────────────────────────────────
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Analisis Sentimen & Pemodelan Topik",
    page_icon="🔍",
    layout="wide",
)

# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f2027 100%);
    color: #e2e8f0;
}

/* Header */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.main-header p {
    color: #94a3b8;
    font-size: 1rem;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Sentiment badge */
.badge-pos {
    display: inline-block;
    background: linear-gradient(90deg, #10b981, #34d399);
    color: #fff;
    font-weight: 700;
    font-size: 1.1rem;
    border-radius: 30px;
    padding: 0.4rem 1.2rem;
}
.badge-neg {
    display: inline-block;
    background: linear-gradient(90deg, #ef4444, #f87171);
    color: #fff;
    font-weight: 700;
    font-size: 1.1rem;
    border-radius: 30px;
    padding: 0.4rem 1.2rem;
}

/* Topic box */
.topic-box {
    background: rgba(129,140,248,0.12);
    border-left: 4px solid #818cf8;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
}
.topic-label {
    font-weight: 600;
    color: #a5b4fc;
    margin-bottom: 0.2rem;
}

/* Tabs */
div[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px;
    padding: 4px;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.1);
}

/* Table */
.dataframe {
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# LOAD MODELS  ← sesuaikan path file model kamu di sini
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    # ── SVM ──
    with open("svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # ── LDA ──
    lda_model = gensim.models.LdaModel.load("lda_model")   # gensim save
    dictionary = corpora.Dictionary.load("lda_dictionary")  # gensim save

    return svm_model, vectorizer, lda_model, dictionary

# ════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_nlp():
    factory_stem   = StemmerFactory()
    stemmer        = factory_stem.create_stemmer()
    factory_stop   = StopWordRemoverFactory()
    stopword_list  = set(factory_stop.get_stop_words())
    return stemmer, stopword_list

# Kamus normalisasi singkatan — tambahkan sesuai dataset kamu
NORM_DICT = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak",
    "yg": "yang", "dgn": "dengan", "udah": "sudah", "sdh": "sudah",
    "bgt": "banget", "bngts": "banget", "klo": "kalau", "kalo": "kalau",
    "aja": "saja", "jg": "juga", "lg": "lagi", "trs": "terus",
    "blm": "belum", "emg": "memang", "tp": "tapi", "dr": "dari",
    "utk": "untuk", "sm": "sama", "mk": "maka", "pd": "pada",
}

def preprocess(text: str, stemmer, stopword_list) -> str:
    # 1. Case folding
    text = text.lower()
    # 2. Data cleaning — hapus mention, URL, hashtag, karakter non-alfabet
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 3. Normalisasi
    tokens = text.split()
    tokens = [NORM_DICT.get(t, t) for t in tokens]
    # 4. Stopword removal
    tokens = [t for t in tokens if t not in stopword_list]
    # 5. Stemming
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens  # kembalikan list token

def tokens_to_str(tokens):
    return " ".join(tokens)

# ════════════════════════════════════════════════════════════════════════════
# PREDICT FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
def predict_sentiment(text_clean_str, svm_model, vectorizer):
    vec = vectorizer.transform([text_clean_str])
    label = svm_model.predict(vec)[0]
    proba = None
    if hasattr(svm_model, "predict_proba"):
        proba = svm_model.predict_proba(vec)[0]
    return label, proba

def get_topics(tokens, lda_model, dictionary, topn_words=10):
    bow = dictionary.doc2bow(tokens)
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
    topic_dist_sorted = sorted(topic_dist, key=lambda x: x[1], reverse=True)
    topics_words = []
    for tid, prob in topic_dist_sorted:
        words = lda_model.show_topic(tid, topn=topn_words)
        topics_words.append({
            "topic_id": tid + 1,
            "probability": round(float(prob), 4),
            "keywords": ", ".join([w for w, _ in words]),
        })
    return topics_words

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🔍 Analisis Sentimen & Pemodelan Topik</h1>
    <p>Support Vector Machine (SVM) + Latent Dirichlet Allocation (LDA)</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════════
with st.spinner("Memuat model..."):
    try:
        svm_model, vectorizer, lda_model, dictionary = load_models()
        stemmer, stopword_list = init_nlp()
        st.success("✅ Model berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.info("Pastikan file model (svm_model.pkl, tfidf_vectorizer.pkl, lda_model, lda_dictionary) ada di folder yang sama dengan app.py")
        st.stop()

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["✏️  Input Teks Langsung", "📂  Upload File CSV"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TEKS TUNGGAL
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Masukkan teks yang ingin dianalisis")
    user_input = st.text_area(
        label="Teks",
        placeholder="Contoh: Pelayanan aplikasi ini sangat lambat dan mengecewakan...",
        height=140,
        label_visibility="collapsed",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyze_btn = st.button("🚀 Analisis", use_container_width=True)

    if analyze_btn:
        if not user_input.strip():
            st.warning("⚠️ Teks tidak boleh kosong.")
        else:
            with st.spinner("Memproses..."):
                tokens  = preprocess(user_input, stemmer, stopword_list)
                clean   = tokens_to_str(tokens)
                label, proba = predict_sentiment(clean, svm_model, vectorizer)
                topics  = get_topics(tokens, lda_model, dictionary)

            st.markdown("---")
            col_sent, col_clean = st.columns(2)

            with col_sent:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Hasil Sentimen (SVM)**")
                badge_cls = "badge-pos" if str(label).lower() in ["positif", "positive", "pos", "1"] else "badge-neg"
                badge_lbl = "😊 Positif" if badge_cls == "badge-pos" else "😞 Negatif"
                st.markdown(f'<span class="{badge_cls}">{badge_lbl}</span>', unsafe_allow_html=True)
                if proba is not None:
                    classes = list(svm_model.classes_)
                    prob_df = pd.DataFrame({"Kelas": classes, "Probabilitas": [f"{p:.2%}" for p in proba]})
                    st.dataframe(prob_df, hide_index=True, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_clean:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Teks Setelah Preprocessing**")
                st.code(clean if clean else "(tidak ada token tersisa)", language=None)
                st.markdown(f"🔢 Jumlah token: **{len(tokens)}**")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### 📊 Distribusi Topik (LDA)")
            top3 = topics[:3]
            cols = st.columns(len(top3))
            for i, (col, t) in enumerate(zip(cols, top3)):
                with col:
                    st.markdown(f"""
                    <div class="topic-box">
                        <div class="topic-label">Topik {t['topic_id']}</div>
                        <div>Probabilitas: <strong>{t['probability']:.4f}</strong></div>
                        <div style="color:#cbd5e1;font-size:0.82rem;margin-top:4px">{t['keywords']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("Lihat semua distribusi topik"):
                topic_df = pd.DataFrame(topics)
                topic_df.columns = ["ID Topik", "Probabilitas", "Kata Kunci"]
                st.dataframe(topic_df, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — UPLOAD CSV
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Upload file CSV")
    st.info("📋 Format CSV: minimal ada kolom **teks** (nama kolom bisa disesuaikan di bawah)", icon="ℹ️")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown(f"**Preview data** ({len(df)} baris)")
            st.dataframe(df.head(5), use_container_width=True)

            text_col = st.selectbox("Pilih kolom teks:", df.columns.tolist())
            max_rows = st.slider("Jumlah baris yang dianalisis:", 1, min(len(df), 500), min(len(df), 100))

            if st.button("🚀 Analisis CSV", use_container_width=False):
                df_sample = df.head(max_rows).copy()
                results = []
                progress = st.progress(0, text="Memproses...")

                for i, row in enumerate(df_sample[text_col].astype(str)):
                    tokens = preprocess(row, stemmer, stopword_list)
                    clean  = tokens_to_str(tokens)
                    label, _ = predict_sentiment(clean, svm_model, vectorizer)
                    topics   = get_topics(tokens, lda_model, dictionary, topn_words=5)
                    dominant = topics[0] if topics else {}
                    results.append({
                        "Teks Asli"       : row[:80] + ("..." if len(row) > 80 else ""),
                        "Sentimen"        : str(label),
                        "Topik Dominan"   : f"Topik {dominant.get('topic_id','?')}",
                        "Prob. Topik"     : dominant.get("probability", 0),
                        "Kata Kunci Topik": dominant.get("keywords", ""),
                    })
                    progress.progress((i + 1) / max_rows, text=f"Memproses baris {i+1}/{max_rows}...")

                progress.empty()
                result_df = pd.DataFrame(results)
                st.success(f"✅ Selesai menganalisis {max_rows} data!")

                # Ringkasan sentimen
                st.markdown("#### 📊 Ringkasan Sentimen")
                sent_count = result_df["Sentimen"].value_counts().reset_index()
                sent_count.columns = ["Sentimen", "Jumlah"]
                sent_count["Persentase"] = (sent_count["Jumlah"] / max_rows * 100).round(2).astype(str) + "%"
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.dataframe(sent_count, hide_index=True, use_container_width=True)
                with col_b:
                    st.bar_chart(sent_count.set_index("Sentimen")["Jumlah"])

                # Hasil lengkap
                st.markdown("#### 📋 Hasil Lengkap")
                st.dataframe(result_df, hide_index=True, use_container_width=True)

                # Download
                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Hasil (CSV)",
                    data=csv_out,
                    file_name="hasil_analisis.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"❌ Error membaca file: {e}")

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#475569;font-size:0.8rem;">Analisis Sentimen & Pemodelan Topik · SVM + LDA · Dibuat untuk keperluan HKI</p>',
    unsafe_allow_html=True,
)
