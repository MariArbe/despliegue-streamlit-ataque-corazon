# -*- coding: utf-8 -*-
import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Latido", page_icon="🫀", layout="wide")

# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=DM+Sans:wght@400;500&display=swap');
html, body, [class*="st-"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, .big-number, .hero-title { font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.02em; }
.block-container { padding-top: 2rem; max-width: 1100px; }
.hero-title { font-size: 3.4rem; font-weight: 800; line-height: 1; margin: 0; }
.hero-sub { font-size: 1.05rem; opacity: .75; margin: .4rem 0 1.2rem 0; max-width: 58ch; }
.verdict { display:flex; align-items:baseline; gap:1.2rem; margin: 1.2rem 0 .4rem 0; flex-wrap:wrap; }
.big-number { font-size: 4.4rem; font-weight: 800; line-height: 1; }
.verdict-text { font-size: 1.15rem; max-width: 42ch; }
.footnote { font-size:.82rem; opacity:.65; margin-top:2rem; }
@media (max-width: 640px) { .hero-title { font-size: 2.4rem; } .big-number { font-size: 3.2rem; } }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Modelo
# 1) Se intenta cargar el .pkl de clase: [modelo, label encoder, variables, scaler]
# 2) Si el servidor tiene otras versiones de numpy/scikit-learn y el pickle no abre,
#    se usa modelo-class.npz: los mismos parámetros del SVM, calculados solo con numpy.
# ---------------------------------------------------------------------------
class _SVMNumpy:
    def __init__(self, z):
        self.sv, self.dual = z["sv"], z["dual"]
        self.b, self.gamma = float(z["intercept"][0]), float(z["gamma"])

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        d2 = ((self.sv[None, :, :] - X[:, None, :]) ** 2).sum(-1)
        return (np.exp(-self.gamma * d2) @ self.dual.T).ravel() + self.b

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


class _LabelNumpy:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes)

    def inverse_transform(self, idx):
        return self.classes_[np.asarray(idx, dtype=int)]


class _ScalerNumpy:
    def __init__(self, scale, min_):
        self.scale, self.min_ = scale, min_

    def transform(self, X):
        return np.asarray(X, dtype=float) * self.scale + self.min_


@st.cache_resource
def cargar():
    try:
        with open("modelo-class.pkl", "rb") as f:
            modelo, le, variables, scaler = pickle.load(f)
        return modelo, le, list(variables), scaler, "pkl"
    except Exception:
        z = np.load("modelo-class.npz")
        return (
            _SVMNumpy(z),
            _LabelNumpy(z["classes"]),
            [str(v) for v in z["variables"]],
            _ScalerNumpy(z["scale"], z["min_"]),
            "npz",
        )


modelo, le, variables, min_max_scaler, origen = cargar()

FUMA = {
    "Nunca ha fumado": "'never smoked'",
    "Fumó antes": "'formerly smoked'",
    "Fuma actualmente": "smokes",
    "Sin información": "Unknown",
}


def preparar(age, glucosa, fuma, hipertension, corazon, casado) -> pd.DataFrame:
    """Mismos pasos del entrenamiento: dummies + normalizar age y glucosa."""
    fila = dict.fromkeys(variables, 0)
    fila["age"] = age
    fila["avg_glucose_level"] = glucosa
    fila[f"smoking_status_{FUMA[fuma]}"] = 1
    fila["hypertension_Yes"] = int(hipertension)
    fila["heart_disease_Yes"] = int(corazon)
    fila["ever_married_Yes"] = int(casado)
    X = pd.DataFrame([fila], columns=variables)
    # En despliegue no se llama fit, solo transform
    X[["age", "avg_glucose_level"]] = min_max_scaler.transform(X[["age", "avg_glucose_level"]])
    return X


def indice(**kw) -> float:
    """El SVM no da probabilidades: convertimos su distancia a la frontera en un
    índice de 0 a 1 (0.5 = justo en la frontera de decisión)."""
    d = float(modelo.decision_function(preparar(**kw))[0])
    return 1 / (1 + np.exp(-3 * d))


# ---------------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Tu perfil")
    age = st.slider("Edad", 18, 82, 45)
    glucosa = st.slider("Glucosa promedio (mg/dL)", 55, 272, 100)
    fuma = st.selectbox("Tabaquismo", list(FUMA.keys()))
    hipertension = st.toggle("Hipertensión")
    corazon = st.toggle("Enfermedad cardíaca previa")
    casado = st.toggle("Ha estado casado/a")

perfil = dict(age=age, glucosa=glucosa, fuma=fuma, hipertension=hipertension, corazon=corazon, casado=casado)
X = preparar(**perfil)
pred = le.inverse_transform(modelo.predict(X))[0]  # 'Yes' / 'No'
alerta = pred == "Yes"
p = indice(**perfil)

if p < 0.3:
    color, mood = "#3DDBC2", "Pulso tranquilo"
elif not alerta:
    color, mood = "#F2A541", "Conviene vigilar"
else:
    color, mood = "#FF4D6D", "Pulso agitado"


# ---------------------------------------------------------------------------
# Electrocardiograma animado
# ---------------------------------------------------------------------------
def ecg_html(p: float, color: str, mood: str) -> str:
    bpm = 62 + 78 * p
    beat_s = 60 / bpm
    rng = np.random.default_rng(7)
    beats = []
    for i in range(8):
        x = i * 100
        amp = 38 * (1 + (rng.random() - 0.5) * 0.9 * p)
        t_dip = 38 - 6 * p * rng.random()
        beats.append(
            f"M{x},50 L{x+15},50 Q{x+20},44 {x+25},50 L{x+36},50 L{x+39},56 "
            f"L{x+43},{50-amp:.1f} L{x+47},64 L{x+50},50 L{x+62},50 "
            f"Q{x+70},{t_dip:.1f} {x+78},50 L{x+100},50"
        )
    d = " ".join(beats)
    return f"""
<div style="background:#0F2A33;border-radius:18px;padding:14px 18px 8px 18px;overflow:hidden;font-family:'DM Sans',sans-serif;">
  <div style="display:flex;justify-content:space-between;align-items:center;color:{color};">
    <span style="font-weight:700;font-size:15px;">{mood}</span>
    <span style="display:flex;align-items:center;gap:8px;font-size:15px;">
      <svg class="heart" width="22" height="22" viewBox="0 0 24 24" fill="{color}">
        <path d="M12 21s-7.5-4.6-9.6-9.2C.9 8.4 2.7 5 6 5c2 0 3.4 1.1 4 2.3h4C14.6 6.1 16 5 18 5c3.3 0 5.1 3.4 3.6 6.8C19.5 16.4 12 21 12 21z"/>
      </svg>
      {bpm:.0f} lpm simulados
    </span>
  </div>
  <svg viewBox="0 0 800 100" width="100%" height="130" preserveAspectRatio="none">
    <defs>
      <pattern id="grid" width="25" height="25" patternUnits="userSpaceOnUse">
        <path d="M25 0H0V25" fill="none" stroke="#1d4350" stroke-width="1"/>
      </pattern>
      <linearGradient id="fade" x1="0" x2="1">
        <stop offset="0" stop-color="#fff" stop-opacity="0"/>
        <stop offset=".15" stop-color="#fff" stop-opacity="1"/>
        <stop offset=".9" stop-color="#fff" stop-opacity="1"/>
        <stop offset="1" stop-color="#fff" stop-opacity="0"/>
      </linearGradient>
      <mask id="m"><rect width="800" height="100" fill="url(#fade)"/></mask>
    </defs>
    <rect width="800" height="100" fill="url(#grid)"/>
    <g mask="url(#m)">
      <g class="trace">
        <path d="{d}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
        <path d="{d}" transform="translate(800)" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
      </g>
    </g>
  </svg>
</div>
<style>
  .trace {{ animation: slide {8*beat_s:.3f}s linear infinite; }}
  @keyframes slide {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-800px); }} }}
  .heart {{ transform-origin: center; animation: beat {beat_s:.3f}s ease-in-out infinite; }}
  @keyframes beat {{ 0%,100% {{ transform: scale(1); }} 15% {{ transform: scale(1.3); }} 30% {{ transform: scale(1); }} }}
  @media (prefers-reduced-motion: reduce) {{ .trace, .heart {{ animation: none; }} }}
</style>
"""


st.markdown('<p class="hero-title">Latido</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Mueve los controles de la izquierda y mira cómo cambia el pulso. '
    "Un modelo SVM entrenado en clase estima si el perfil se parece al de personas que sufrieron un ataque al corazón.</p>",
    unsafe_allow_html=True,
)
st.iframe(ecg_html(p, color, mood), height=215)

msg = (
    "El modelo clasifica este perfil como de riesgo. Vale la pena consultarlo con un profesional de salud."
    if alerta
    else "El modelo no clasifica este perfil como de riesgo."
)
num_color = "#1B998B" if color == "#3DDBC2" else color
st.markdown(
    f'<div class="verdict"><span class="big-number" style="color:{num_color}">{p*100:.0f}</span>'
    f'<span class="verdict-text"><b>Índice de riesgo (0 a 100)</b><br>{msg}</span></div>',
    unsafe_allow_html=True,
)
st.progress(p, text="A partir de 50 el modelo clasifica como riesgo")

# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["¿Qué mueve el resultado?", "Curvas", "Sobre el modelo"])

with tab1:
    st.write(
        "Para cada variable, cuánto cambia el índice si pasa de su valor más bajo al más alto "
        "mientras lo demás queda como lo configuraste."
    )
    extremos = {
        "Edad": ("age", 18, 82),
        "Glucosa": ("glucosa", 55, 272),
        "Tabaquismo": ("fuma", "Nunca ha fumado", "Fuma actualmente"),
        "Hipertensión": ("hipertension", False, True),
        "Enfermedad cardíaca": ("corazon", False, True),
        "Ha estado casado/a": ("casado", False, True),
    }
    filas = []
    for etiqueta, (k, lo, hi) in extremos.items():
        a, b = dict(perfil), dict(perfil)
        a[k], b[k] = lo, hi
        filas.append({"Variable": etiqueta, "Cambio": (indice(**b) - indice(**a)) * 100})
    df = pd.DataFrame(filas).sort_values("Cambio", key=abs, ascending=False)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Cambio:Q", title="Cambio en el índice (puntos)"),
            y=alt.Y("Variable:N", sort=None, title=None),
            color=alt.condition(alt.datum.Cambio > 0, alt.value("#E63462"), alt.value("#1B998B")),
            tooltip=["Variable", alt.Tooltip("Cambio:Q", format="+.1f")],
        )
        .properties(height=38 * len(df) + 30)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")

with tab2:
    st.write(
        "Así cambia el índice si solo varías una cosa y dejas el resto igual. "
        "El punto marca dónde estás ahora."
    )
    c1, c2 = st.columns(2)

    def curva(col, clave, valores, actual, titulo):
        dfc = pd.DataFrame({titulo: valores, "Índice": [indice(**{**perfil, clave: v}) * 100 for v in valores]})
        linea = alt.Chart(dfc).mark_line(color="#0F2A33", strokeWidth=3).encode(
            x=alt.X(f"{titulo}:Q", title=titulo), y=alt.Y("Índice:Q", scale=alt.Scale(domain=[0, 100]))
        )
        punto = alt.Chart(pd.DataFrame({titulo: [actual], "Índice": [p * 100]})).mark_point(
            color=num_color, size=160, filled=True
        ).encode(x=f"{titulo}:Q", y="Índice:Q")
        col.altair_chart((linea + punto).properties(height=240), width="stretch")

    curva(c1, "age", list(range(18, 83, 2)), age, "Edad")
    curva(c2, "glucosa", list(range(55, 273, 5)), glucosa, "Glucosa")
    st.caption("Son asociaciones que aprendió el modelo con los datos, no efectos causales.")

with tab3:
    st.write(
        "Modelo: SVM (SVC con kernel RBF y `class_weight='balanced'`), entrenado con edad, glucosa, "
        "tabaquismo, hipertensión, enfermedad cardíaca y estado civil. Edad y glucosa van normalizadas "
        "con MinMaxScaler."
    )
    if origen == "npz":
        st.caption("Modelo cargado desde la copia portable (modelo-class.npz).")
    st.write(
        "Un SVM no entrega probabilidades. El índice se calcula a partir de la distancia del perfil a la "
        "frontera de decisión, así que sirve para comparar perfiles, pero no es un porcentaje de probabilidad real."
    )

st.markdown(
    '<p class="footnote">Proyecto académico. Esta herramienta no es un diagnóstico médico ni '
    "reemplaza la opinión de un profesional de salud.</p>",
    unsafe_allow_html=True,
)
