import streamlit as st

st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Predição de Predisposição à Obesidade")
st.write(
    """
    Preencha os dados abaixo para avaliar a **predisposição à obesidade**
    com base em um modelo de **Machine Learning (XGBoost)**.
    """
)

st.divider()

with st.form("form_obesidade"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Sexo", ["Male", "Female"])
        age = st.slider("Idade", 10, 80, 30)
        height = st.number_input("Altura (m)", min_value=1.20, max_value=2.20, value=1.70)
        weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
        family_history = st.selectbox(
            "Histórico familiar de obesidade?",
            ["yes", "no"]
        )

    with col2:
        favc = st.selectbox(
            "Consome alimentos calóricos frequentemente?",
            ["yes", "no"]
        )
        fcvc = st.slider(
            "Consumo de vegetais (1 = baixo | 3 = alto)",
            1, 3, 2
        )
        ncp = st.slider(
            "Número de refeições principais por dia",
            1, 4, 3
        )
        caec = st.selectbox(
            "Consumo alimentar entre refeições",
            ["no", "Sometimes", "Frequently", "Always"]
        )
        smoke = st.selectbox(
            "Fuma?",
            ["yes", "no"]
        )

    col3, col4 = st.columns(2)

    with col3:
        ch2o = st.slider(
            "Consumo diário de água (1 = baixo | 3 = alto)",
            1, 3, 2
        )
        scc = st.selectbox(
            "Monitora consumo calórico?",
            ["yes", "no"]
        )
        faf = st.slider(
            "Atividade física semanal (0 = nenhuma | 3 = alta)",
            0, 3, 1
        )

    with col4:
        tue = st.slider(
            "Tempo de uso de tecnologia (0 = baixo | 2 = alto)",
            0, 2, 1
        )
        calc = st.selectbox(
            "Consumo de álcool",
            ["no", "Sometimes", "Frequently", "Always"]
        )
        mtrans = st.selectbox(
            "Meio de transporte",
            ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
        )

    submit = st.form_submit_button("🔍 Avaliar Predisposição")


import pandas as pd
import joblib


@st.cache_resource
def load_model():
    return joblib.load("modelo_xgb_obesidade.joblib")

model = load_model()


# ===============================
# PREDIÇÃO
# ===============================
if submit:

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }])

    # 🔹 Predição
    prob = model.predict_proba(input_data)[0][1]
    pred = int(prob >= 0.55)

    st.divider()
    st.subheader("📊 Resultado da Avaliação")

    st.metric(
        label="Probabilidade de Predisposição à Obesidade",
        value=f"{prob:.2%}"
    )

    if pred == 1:
        st.error(
            "⚠️ **Alta predisposição à obesidade**\n\n"
            "Recomenda-se acompanhamento nutricional e hábitos mais saudáveis."
        )
    else:
        st.success(
            "✅ **Baixa predisposição à obesidade**\n\n"
            "Continue mantendo hábitos saudáveis!"
        )

