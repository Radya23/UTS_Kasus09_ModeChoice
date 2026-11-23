import os
import joblib
import pandas as pd
import streamlit as st
from fitur import buat_fitur_baru, MODES

st.set_page_config(page_title="Prediksi Pilihan Moda Transportasi", layout="wide")

MODEL_PATH = os.path.join("models", "model_mode_choice.joblib")

@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

artifact = load_artifact()

st.title("Prediksi Pilihan Moda Transportasi (Kasus 9)")
st.markdown(
    "Aplikasi klasifikasi multikelas untuk memprediksi pilihan moda transportasi "
    "(**Moda 1 – 4**) berdasarkan karakteristik perjalanan dan atribut keempat alternatif moda."
)

if artifact is None:
    st.error("File model tidak ditemukan. Jalankan `python latih.py` terlebih dahulu.")
    st.stop()

pipeline = artifact["pipeline"]
default_sample = artifact["default_sample"]

tab_pred, tab_eval = st.tabs(["Simulasi Prediksi", "Metrik & Evaluasi Model"])

with tab_pred:
    st.subheader("1. Profil Pelancong")
    col_ind1, col_ind2 = st.columns(2)
    with col_ind1:
        hinc = st.number_input(
            "Pendapatan Rumah Tangga (hinc)",
            min_value=0.0, max_value=200.0,
            value=float(default_sample.get("hinc", 35.0)), step=1.0
        )
    with col_ind2:
        psize = st.number_input(
            "Ukuran Rombongan (psize)",
            min_value=1, max_value=15,
            value=int(default_sample.get("psize", 1)), step=1
        )

    st.subheader("2. Atribut Keempat Alternatif Moda")
    cols = st.columns(4)
    input_data = {"hinc": hinc, "psize": float(psize)}

    for idx, m in enumerate(MODES):
        with cols[idx]:
            st.markdown(f"**Moda {m}**")
            input_data[f"ttme_mode{m}"] = st.number_input(
                f"Waktu Terminal (ttme) - M{m}",
                min_value=0.0, max_value=300.0,
                value=float(default_sample.get(f"ttme_mode{m}", 30.0)), step=1.0
            )
            input_data[f"invc_mode{m}"] = st.number_input(
                f"Biaya Dalam Kendaraan (invc) - M{m}",
                min_value=0.0, max_value=500.0,
                value=float(default_sample.get(f"invc_mode{m}", 30.0)), step=1.0
            )
            input_data[f"invt_mode{m}"] = st.number_input(
                f"Waktu Dalam Kendaraan (invt) - M{m}",
                min_value=0.0, max_value=2000.0,
                value=float(default_sample.get(f"invt_mode{m}", 200.0)), step=5.0
            )
            input_data[f"gc_mode{m}"] = st.number_input(
                f"Biaya Gabungan (gc) - M{m}",
                min_value=0.0, max_value=1000.0,
                value=float(default_sample.get(f"gc_mode{m}", 70.0)), step=1.0
            )

    if st.button("Prediksi Pilihan Moda", type="primary"):
        df_input_raw = pd.DataFrame([input_data])
        df_input_feat = buat_fitur_baru(df_input_raw)
        # Pastikan urutan kolom persis sama seperti saat training
        df_input_feat = df_input_feat[artifact["engineered_features"]]

        pred_class = int(pipeline.predict(df_input_feat)[0])
        proba = pipeline.predict_proba(df_input_feat)[0]
        classes = artifact["classes"]

        st.success(f"Hasil Prediksi: Pelancong diprediksi memilih **Moda {pred_class}** (Probabilitas: {max(proba)*100:.2f}%)")

        proba_df = pd.DataFrame({
            "Moda": [f"Moda {c}" for c in classes],
            "Probabilitas": proba
        }).set_index("Moda")

        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.dataframe(proba_df.style.format({"Probabilitas": "{:.2%}"}), use_container_width=True)
        with col_res2:
            st.bar_chart(proba_df)

with tab_eval:
    st.subheader(f"Perbandingan Cross-Validation & Evaluasi Test ({artifact['best_model_name']})")

    cv_df = pd.DataFrame(artifact["cv_results"]).T
    cv_df.columns = ["CV Accuracy (Mean)", "CV Accuracy (Std)", "CV Macro F1 (Mean)", "CV Macro F1 (Std)"]
    st.markdown("**1. Perbandingan Model pada 5-Fold Stratified Cross-Validation (Data Train)**")
    st.dataframe(cv_df.style.format("{:.4f}"), use_container_width=True)

    test_m = artifact["test_metrics"]
    st.markdown("**2. Performa Evaluasi pada Data Test (1x Evaluasi)**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test Accuracy", f"{test_m['accuracy']:.2%}")
    m2.metric("Test Macro F1-Score", f"{test_m['f1_macro']:.4f}")
    m3.metric("Test Weighted F1-Score", f"{test_m['f1_weighted']:.4f}")

    st.markdown("**3. Confusion Matrix (Data Test)**")
    cm_df = pd.DataFrame(
        test_m["confusion_matrix"],
        index=[f"Aktual Moda {c}" for c in artifact["classes"]],
        columns=[f"Prediksi Moda {c}" for c in artifact["classes"]]
    )
    st.dataframe(cm_df, use_container_width=True)
