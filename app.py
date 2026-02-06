import streamlit as st
import joblib
import numpy as np
import os

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="Membrane Performance Prediction", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    h1 { color: #1e3a8a; text-align: center; margin-bottom: 20px; }
    .stNumberInput > label { font-size: 15px; font-weight: 600; color: #374151; }
    div[data-testid="stExpander"] { background-color: white; border-radius: 10px; }
    .stButton > button { 
        width: 100%; 
        background-color: #2563eb; 
        color: white; 
        font-size: 18px; 
        padding: 12px; 
        border-radius: 8px; 
        border: none;
    }
    .stButton > button:hover { background-color: #1d4ed8; }
    /* 增加结果展示卡片的样式 */
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Membrane Performance Prediction")
st.markdown("<h4 style='text-align: center; color: gray;'>Prediction of Rejection & Permeability</h4>", unsafe_allow_html=True)


# ==========================================
# 2. 加载两个模型
# ==========================================
@st.cache_resource
def load_models():
    # 路径定义
    path_rejection = 'models/cat.joblib'
    path_Permeability = 'models/gbm.joblib'

    models = {}

    # 加载截留率模型
    if os.path.exists(path_rejection):
        models['rejection'] = joblib.load(path_rejection)
    else:
        models['rejection'] = None

    # 加载通量模型
    if os.path.exists(path_Permeability):
        models['Permeability'] = joblib.load(path_Permeability)
    else:
        models['Permeability'] = None

    return models


# 加载模型
loaded_models = load_models()
model_rejection = loaded_models['rejection']
model_Permeability = loaded_models['Permeability']

# 检查模型状态
if model_rejection is None:
    st.error("❌ ERROR: 'models/cat.joblib' (Rejection Model) not found!")
if model_Permeability is None:
    st.warning("⚠️ WARNING: 'models/a.joblib' (Permeability Model) not found! Permeability prediction will be unavailable.")

if model_rejection is None and model_Permeability is None:
    st.stop()  # 如果两个都没找到，停止运行

# ==========================================
# 3. 数值输入表格
# ==========================================
st.markdown("### 📋 Please enter membrane parameter indicators")

col1, col2 = st.columns(2)  # 调整为两列布局看起来更整齐

with col1:
    st.markdown("#### Membrane Parameters")
    ca = st.number_input("Contact angle (°)", value=15.0, step=1.0, format="%.2f")
    prr = st.number_input("Pore radius rp (nm)", value=1.0, step=1.0, format="%.2f")
    mwco = st.number_input("MWCO (Da)", value=1.0, step=1.0, format="%.2f")
    rms = st.number_input("RMS roughness (nm)", value=45.00, step=1.0, format="%.2f")
    # Film thickness (仅用于截留率预测)
    ft = st.number_input("Film thickness (nm)", value=1.0, step=1.0, format="%.3f", help="Used for Rejection only")

with col2:
    st.markdown("#### Operating Conditions")
    p = st.number_input("Pressure (bar)", value=1.00, step=1.0, format="%.2f")
    temp = st.number_input("Temperature (°C)", value=500.0, step=1.0, format="%.1f")
    smw = st.number_input("Solute MW (g/mol)", value=100.0, step=1.0, format="%.1f")
    # Concentration (仅用于截留率预测)
    conc = st.number_input("Concentration (mg/L)", value=1.0, step=1.0, format="%.3f", help="Used for Rejection only")

# ==========================================
# 4. 预测逻辑
# ==========================================
st.markdown("---")

if st.button("Start Prediction"):
    try:
        # ---------------------------------------------------
        # 1. 截留率 (Rejection) 数据准备 - 使用全部9个参数
        # 顺序: [ca, prr, mwco, rms, ft, p, temp, conc, smw]
        # ---------------------------------------------------
        input_vector_rejection = [ca, prr, mwco, rms, ft, p, temp, conc, smw]

        # ---------------------------------------------------
        # 2. 通量 (Permeability) 数据准备 - 剔除 ft 和 conc
        # 剩余参数: ca, prr, mwco, rms, p, temp, smw
        # ⚠️注意：这里假设 a.joblib 训练时的特征顺序就是下面这个顺序。
        # 如果训练顺序不同，请调整下面列表中的变量位置。
        # ---------------------------------------------------
        input_vector_Permeability = [ca, prr, mwco, rms, p, temp, smw]

        # 转换为模型接受的格式 (2D array)
        final_input_rejection = np.array(input_vector_rejection).reshape(1, -1)
        final_input_Permeability = np.array(input_vector_Permeability).reshape(1, -1)

        # ---------------------------------------------------
        # 3. 执行预测
        # ---------------------------------------------------

        # 结果展示区
        st.markdown("### Prediction Results")
        res_col1, res_col2 = st.columns(2)

        # 预测截留率
        if model_rejection:
            pred_rej = model_rejection.predict(final_input_rejection)[0]
            with res_col1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3 style="color: #4B5563;">Rejection Prediction</h3>
                        <h1 style="color: #2563eb;">{pred_rej:.4f}</h1>
                        <p style="color: gray;">Model: cat.joblib</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            with res_col1:
                st.error("Rejection model not loaded.")

        # 预测通量
        if model_Permeability:
            pred_Permeability = model_Permeability.predict(final_input_Permeability)[0]
            with res_col2:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3 style="color: #4B5563;">Permeability Prediction</h3>
                        <h1 style="color: #059669;">{pred_Permeability:.4f}</h1>
                        <p style="color: gray;">Model: gbm.joblib</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            with res_col2:
                st.error("Permeability model (gbm.joblib) not loaded.")

        # ---------------------------------------------------
        # 4. 调试信息 (Debug)
        # ---------------------------------------------------
        with st.expander("Show Debug Input Data"):
            st.write("**Rejection Input (9 features):**")
            st.code(str(input_vector_rejection))
            st.caption("Order: [ca, prr, mwco, rms, ft, p, temp, conc, smw]")

            st.write("**Permeability Input (7 features - No ft/conc):**")
            st.code(str(input_vector_Permeability))
            st.caption("Order: [ca, prr, mwco, rms, p, temp, smw]")

    except Exception as e:
        st.error(f"Prediction Error: {e}")