import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Flame in Freefall | Hybrid AI Dashboard", layout="wide")

# --- 1. БАЗА ДАНИХ ---
@st.cache_data
def load_data():
    data = [
        {"id": "saffire-scale", "title": "Spacecraft fire growth cannot be extrapolated linearly from small-scale tests", 
         "impact": 5, "actionability": 3, "novelty": 3, "evidence": 5, "iss_fit": 5, "lunar_fit": 4, "mars_fit": 5, "source": "Saffire-I/II/III"},
        {"id": "saffire1-fabric", "title": "Saffire-1: steady flame spread over a large fabric sample in microgravity", 
         "impact": 4, "actionability": 3, "novelty": 2, "evidence": 5, "iss_fit": 5, "lunar_fit": 3, "mars_fit": 4, "source": "Saffire-I"},
        {"id": "saffire2-silicone", "title": "Saffire-2: a silicone sample did not sustain spreading flame after ignition ended", 
         "impact": 4, "actionability": 3, "novelty": 4, "evidence": 5, "iss_fit": 5, "lunar_fit": 3, "mars_fit": 4, "source": "Saffire-II"},
        {"id": "flex-cool-flames", "title": "Droplet flames in microgravity can leave an invisible cool flame", 
         "impact": 3, "actionability": 2, "novelty": 5, "evidence": 5, "iss_fit": 3, "lunar_fit": 2, "mars_fit": 2, "source": "FLEX/CFI"},
        {"id": "gap-partial-gravity", "title": "Predicting burning at lunar and Mars gravity from 1g and 0g data", 
         "impact": 5, "actionability": 1, "novelty": 4, "evidence": 1, "iss_fit": 1, "lunar_fit": 5, "mars_fit": 5, "source": "Task Book (Gap)"}
    ]
    return pd.DataFrame(data)

df = load_data()

# --- 2. БІЧНА ПАНЕЛЬ ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=100)
st.sidebar.header("⚙️ Умови місії та ваги")

mission_type = st.sidebar.selectbox("Цільова місія:", ["ISS / LEO (iss_fit)", "Lunar Surface (lunar_fit)", "Mars Surface (mars_fit)"])
mission_col = mission_type.split(" ")[-1].replace("(", "").replace(")", "")

st.sidebar.markdown("### Ваги критеріїв ранжування (0-100)")
w_impact = st.sidebar.slider("Impact (Вплив на безпеку)", 0, 100, 35)
w_evidence = st.sidebar.slider("Evidence (Надійність даних)", 0, 100, 25)
w_mission = st.sidebar.slider("Mission Fit (Відповідність місії)", 0, 100, 20)
w_action = st.sidebar.slider("Actionability (Можливість застосування)", 0, 100, 15)
w_novelty = st.sidebar.slider("Novelty (Новизна для 0g)", 0, 100, 5)

def calculate_scores(df, w_impact, w_evidence, w_mission, w_action, w_novelty, mission_col):
    total_w = w_impact + w_evidence + w_mission + w_action + w_novelty
    if total_w == 0: total_w = 1
    nw = [w/total_w for w in [w_impact, w_evidence, w_mission, w_action, w_novelty]]
    
    scores = []
    for _, row in df.iterrows():
        vals = [row['impact'], row['evidence'], row[mission_col], row['actionability'], row['novelty']]
        score = sum(nw[i] * ((vals[i] - 1) / 4) for i in range(5)) * 100
        scores.append(round(score, 1))
    
    df['Final_Score'] = scores
    return df.sort_values('Final_Score', ascending=False)

ranked_df = calculate_scores(df.copy(), w_impact, w_evidence, w_mission, w_action, w_novelty, mission_col)

# --- 3. ІНТЕРФЕЙС ---
st.title("🔥 Flame in Freefall: Hybrid AI Dashboard")
st.markdown("Інтеграція динамічного ранжування матеріалів, математичного сурогату горіння та локального автономного ШІ.")

tab1, tab2, tab3 = st.tabs(["📈 Flammability Map", "📊 Evidence Ranking", "🤖 Autonomous AI Agent (RAG)"])

with tab1:
    st.subheader("Теплова карта ймовірності стійкого горіння")
    st.markdown("Змінюйте фізичні параметри середовища, щоб побачити, як зміщується пік пожежної небезпеки.")
    
    # НОВІ ПОВЗУНКИ ДЛЯ ФІЗИКИ
    col1, col2 = st.columns(2)
    with col1:
        pressure = st.slider("Тиск у модулі (кПа)", 50.0, 105.0, 101.3, step=0.1, help="101.3 - МКС, ~70.0 - перспективні місії")
    with col2:
        gravity = st.slider("Гравітація (g)", 0.0, 1.0, 0.0, step=0.01, help="0g - мікрогравітація, 0.16g - Місяць, 0.38g - Марс, 1g - Земля")
    
    o2_grid = np.linspace(15.0, 40.0, 50)
    flow_grid = np.logspace(np.log10(0.5), np.log10(200.0), 50)
    O2, FLOW = np.meshgrid(o2_grid, flow_grid)
    
    # ДИНАМІЧНА МАТЕМАТИКА
    # Зниження тиску вимагає більше кисню для підтримки горіння
    o2_threshold = 14.5 + (101.3 - pressure) * 0.08 
    # Гравітація створює природну конвекцію, що зміщує коридор швидкості потоку
    gravity_shift = gravity * 1.5 
    
    risk_z = np.clip((O2 - o2_threshold) / 25.0 * np.exp(-((np.log10(FLOW) - 1.0 - gravity_shift)**2) / (0.8 + gravity*2 + (O2-15)/50)), 0, 1) * 100
    
    fig_map = go.Figure(data=go.Heatmap(
        z=risk_z, x=o2_grid, y=flow_grid,
        colorscale='Inferno', zmin=0, zmax=100,
        colorbar=dict(title="Ймовірність (%)")
    ))
    fig_map.update_layout(
        xaxis_title="Концентрація кисню O₂ (%)",
        yaxis_title="Швидкість потоку (см/с)",
        yaxis_type="log",
        height=500, margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    st.subheader(f"Ранжування знахідок для {mission_type}")
    
    fig_bar = px.bar(
        ranked_df, x="Final_Score", y="title", color="Final_Score",
        orientation='h', color_continuous_scale="Reds",
        labels={"Final_Score": "Комплексний індекс ризику", "title": "Експеримент/Знахідка"}
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.dataframe(
        ranked_df[['Final_Score', 'source', 'title', 'impact', 'evidence']],
        use_container_width=True, hide_index=True
    )

with tab3:
    st.subheader("💬 Локальний ШІ-асистент безпеки екіпажу")
    st.info(
        "🚀 **Архітектурне рішення:** У реальних умовах глибокого космосу немає стабільного зв'язку для виклику хмарних API. "
        "Цей асистент спроєктований для 100% автономної роботи на локальних бортових серверах."
    )
    
    user_q = st.chat_input("Запитайте про вогнестійкість силікону, бавовни або швидкість поширення вогню...")
    
    with st.chat_message("assistant"):
        st.write("Термінал безпеки активний. Я проаналізував дані експериментів Saffire та FLEX. Чим можу допомогти?")
        
    if user_q:
        with st.chat_message("user"):
            st.write(user_q)
        
        with st.chat_message("assistant"):
            if "silicone" in user_q.lower() or "силікон" in user_q.lower():
                st.write("**RAG Response (Source: Saffire-II):** У мікрогравітації силікон після припинення зовнішнього підпалу залишив лише крихітне газове полум'я і самозгас. Його земні рейтинги безпеки не збігаються з космічними.")
            else:
                st.write("Це MVP-демо. У продакшені сюди підключається локальний LLM через Ollama або MQTT-інтерфейс.")