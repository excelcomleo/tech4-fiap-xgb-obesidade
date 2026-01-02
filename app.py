import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuração da página com tema moderno
st.set_page_config(
    page_title="Check-up de Saúde: Predisposição",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização Customizada (CSS)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .metric-container {
        text-align: center;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Carregar modelo e metadados
@st.cache_resource
def load_model():
    model = joblib.load("modelo_xgb_obesidade.joblib")
    metadata = joblib.load("model_metadata.joblib")
    return model, metadata

try:
    model, metadata = load_model()
except Exception as e:
    st.error("Erro ao carregar o sistema. Verifique se os arquivos .joblib estão na mesma pasta.")
    st.stop()

# --- HEADER ---
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.title("Calculadora de Tendência à Saúde")
    st.markdown("#### Entenda sua predisposição genética e comportamental para uma vida mais equilibrada.")

st.divider()

# --- SIDEBAR / INPUTS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
st.sidebar.header("Sobre Você")

def user_input_features():
    # Tradução de termos técnicos para perguntas amigáveis
    
    with st.sidebar:
        st.subheader("👤 Perfil Básico")
        gender = st.selectbox("Qual seu gênero?", ["Feminino", "Masculino"], index=0)
        gender_map = {"Feminino": "Female", "Masculino": "Male"}
        
        age = st.number_input("Qual sua idade?", 14, 90, 25)
        height = st.number_input("Qual sua altura? (em metros)", 1.20, 2.20, 1.70, step=0.01)
        
        st.subheader("🧬 Histórico")
        family = st.selectbox("Existem casos de sobrepeso na sua família?", ["Sim", "Não"])
        family_map = {"Sim": "yes", "Não": "no"}
        
        st.subheader("🍎 Hábitos Alimentares")
        favc = st.selectbox("Você costuma comer alimentos muito calóricos com frequência?", ["Sim", "Não"])
        favc_map = {"Sim": "yes", "Não": "no"}
        
        fcvc = st.select_slider(
            "Com que frequência você come vegetais nas refeições?",
            options=[1, 2, 3],
            value=2,
            help="1: Raramente, 2: Às vezes, 3: Sempre"
        )
        
        ncp = st.select_slider(
            "Quantas refeições principais você faz por dia?",
            options=[1, 2, 3, 4],
            value=3
        )
        
        caec = st.selectbox(
            "Você costuma 'beliscar' entre as refeições?",
            ["Não", "Às vezes", "Frequentemente", "Sempre"]
        )
        caec_map = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
        
        ch2o = st.select_slider(
            "Quantos litros de água você bebe por dia?",
            options=[1, 2, 3],
            value=2,
            help="1: Menos de 1L, 2: Entre 1L e 2L, 3: Mais de 2L"
        )
        
        st.subheader("🏃 Estilo de Vida")
        faf = st.select_slider(
            "Quantos dias por semana você pratica atividade física?",
            options=[0, 1, 2, 3],
            value=1,
            help="0: Nenhum, 1: 1 a 2 dias, 2: 2 a 4 dias, 3: Mais de 4 dias"
        )
        
        tue = st.select_slider(
            "Quanto tempo você passa em frente a telas (celular/TV) por dia?",
            options=[0, 1, 2],
            value=1,
            help="0: 0-2h, 1: 3-5h, 2: Mais de 5h"
        )
        
        calc = st.selectbox("Com que frequência você consome álcool?", ["Não", "Às vezes", "Frequentemente"])
        calc_map = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently"}
        
        smoke = st.selectbox("Você fuma?", ["Sim", "Não"], index=1)
        smoke_map = {"Sim": "yes", "Não": "no"}
        
        scc = st.selectbox("Você costuma contar as calorias do que come?", ["Sim", "Não"], index=1)
        scc_map = {"Sim": "yes", "Não": "no"}
        
        mtrans = st.selectbox(
            "Qual seu principal meio de transporte?",
            ["Transporte Público", "Carro", "Caminhada", "Bicicleta", "Moto"]
        )
        mtrans_map = {
            "Transporte Público": "Public_Transportation",
            "Carro": "Automobile",
            "Caminhada": "Walking",
            "Bicicleta": "Bike",
            "Moto": "Motorbike"
        }

    # Criar dicionário com nomes originais das colunas para o modelo
    data = {
        'Gender': gender_map[gender],
        'Age': age,
        'Height': height,
        'family_history': family_map[family],
        'FAVC': favc_map[favc],
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec_map[caec],
        'SMOKE': smoke_map[smoke],
        'CH2O': ch2o,
        'SCC': scc_map[scc],
        'FAF': faf,
        'TUE': tue,
        'CALC': calc_map[calc],
        'MTRANS': mtrans_map[mtrans]
    }
    return pd.DataFrame([data])

# --- MAIN CONTENT ---
input_df = user_input_features()

col_info, col_res = st.columns([1, 1])

with col_info:
    st.markdown("""
    ### 📋 Como funciona?
    Nosso algoritmo analisa seus hábitos diários e histórico familiar para identificar padrões que podem indicar uma 
    maior ou menor predisposição ao desenvolvimento de obesidade.
    
    **Importante:** Este modelo **não utiliza seu peso atual**, focando apenas em comportamento e genética.
    
    ---
    #### 💡 Dicas para um bom resultado:
    - Seja honesto sobre seus hábitos.
    - O resultado é uma estimativa baseada em dados estatísticos.
    - Consulte sempre um profissional de saúde.
    """)

with col_res:
    st.markdown("### 🚀 Resultado da Análise")
    if st.button("Analisar minha Predisposição"):
        # Predição
        prob = model.predict_proba(input_df)[0][1]
        threshold = 0.55
        is_predisposed = prob >= threshold
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        # Exibição da Probabilidade
        st.write(f"**Probabilidade Calculada:**")
        st.progress(float(prob))
        
        if is_predisposed:
            st.error(f"### ⚠️ Atenção: Tendência Elevada ({prob*100:.1f}%)")
            st.markdown("""
                Seus hábitos e histórico sugerem uma predisposição maior. 
                Pequenas mudanças na rotina podem fazer uma grande diferença a longo prazo!
            """)
        else:
            st.success(f"### ✅ Ótimo: Tendência Baixa ({prob*100:.1f}%)")
            st.markdown("""
                Seus padrões atuais indicam um baixo risco de desenvolvimento de obesidade. 
                Continue mantendo um estilo de vida equilibrado!
            """)
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Preencha os dados ao lado e clique no botão acima para ver o resultado.")

# Rodapé
st.markdown("---")
st.caption("Trabalho de Pós-Graduação em Ciência de Dados & Machine Learning | Modelo: XGBoost")
