import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------------
# Page config + styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .block-container { padding-top: 1.5rem; max-width: 1250px; }

    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        padding: 2.2rem 2.5rem; border-radius: 22px; color: white;
        box-shadow: 0 12px 30px rgba(124,58,237,.35); margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2.3rem; font-weight: 700; color: white; }
    .hero p  { margin: .4rem 0 0 0; font-size: 1.05rem; opacity: .92; }

    .card {
        background: rgba(255,255,255,.04);
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px; padding: 1.2rem 1.4rem; height: 100%;
    }
    .section-title { font-weight: 600; font-size: 1.15rem; margin-bottom: .6rem; }

    .metric-box {
        text-align: center; padding: 1rem .5rem; border-radius: 16px;
        background: linear-gradient(135deg, rgba(79,70,229,.15), rgba(219,39,119,.15));
        border: 1px solid rgba(128,128,128,.25);
    }
    .metric-box .val { font-size: 1.8rem; font-weight: 700; }
    .metric-box .lbl { font-size: .8rem; opacity: .75; text-transform: uppercase; letter-spacing: .06em; }

    .grade-pill {
        display: inline-block; padding: .45rem 1.2rem; border-radius: 999px;
        font-weight: 600; font-size: 1.05rem; color: white;
    }
    .tip {
        padding: .7rem 1rem; border-radius: 12px; margin-bottom: .55rem;
        border-left: 5px solid; background: rgba(128,128,128,.08); font-size: .93rem;
    }
    .tip.good { border-color: #22c55e; }
    .tip.warn { border-color: #f59e0b; }

    div.stButton > button {
        width: 100%; border: none; border-radius: 14px; padding: .8rem 1rem;
        font-weight: 600; font-size: 1.05rem; color: white;
        background: linear-gradient(135deg, #4f46e5, #db2777);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 20px rgba(219,39,119,.4); color: white;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Data + model (same preprocessing & features as the notebook)
# ----------------------------------------------------------------------------
FEATURES = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Gender",
    "Access_to_Resources", "Tutoring_Sessions", "Sleep_Hours", "Previous_Scores",
    "Motivation_Level", "Internet_Access", "Family_Income", "Peer_Influence",
    "Physical_Activity", "Learning_Disabilities",
]

NICE_NAMES = {
    "Hours_Studied": "Hours studied", "Attendance": "Attendance",
    "Parental_Involvement": "Parental involvement", "Gender": "Gender",
    "Access_to_Resources": "Access to resources", "Tutoring_Sessions": "Tutoring sessions",
    "Sleep_Hours": "Sleep hours", "Previous_Scores": "Previous scores",
    "Motivation_Level": "Motivation level", "Internet_Access": "Internet access",
    "Family_Income": "Family income", "Peer_Influence": "Peer influence",
    "Physical_Activity": "Physical activity", "Learning_Disabilities": "Learning disability",
}

LMH = {"Low": 0, "Medium": 1, "High": 2}
MAPPINGS = {
    "Parental_Involvement": LMH, "Access_to_Resources": LMH,
    "Motivation_Level": LMH, "Family_Income": LMH,
    "Internet_Access": {"No": 0, "Yes": 1},
    "Gender": {"Male": 0, "Female": 1},
    "Learning_Disabilities": {"No": 0, "Yes": 1},
    "Peer_Influence": {"Negative": 0, "Neutral": 1, "Positive": 2},
}


@st.cache_data(show_spinner=False)
def prepare(df: pd.DataFrame):
    df = df.copy()
    for col, mp in MAPPINGS.items():
        df[col] = df[col].map(mp)
    df = df.dropna(subset=FEATURES + ["Exam_Score"])
    return df[FEATURES], df["Exam_Score"]


@st.cache_resource(show_spinner="Training model…")
def train(df: pd.DataFrame):
    x, y = prepare(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": r2_score(y_test, pred),
        "train_r2": model.score(x_train, y_train),
    }
    return model, metrics, x, y, y_test, pred


def load_dataframe():
    path = "StudentPerformanceFactors.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    st.sidebar.warning("`StudentPerformanceFactors.csv` not found next to app.py.")
    up = st.sidebar.file_uploader("Upload the dataset (CSV)", type="csv")
    return pd.read_csv(up) if up else None


def grade_for(score):
    if score >= 80: return "A", "Excellent", "#16a34a"
    if score >= 70: return "B", "Good", "#2563eb"
    if score >= 60: return "C", "Average", "#d97706"
    if score >= 50: return "D", "Below average", "#ea580c"
    return "F", "At risk", "#dc2626"


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Performance Predictor</h1>
        <p>Enter a student's habits and background — get an instant exam score prediction,
        see what's helping or hurting it, and explore what-if scenarios.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df_raw = load_dataframe()
if df_raw is None:
    st.info("👈 Upload `StudentPerformanceFactors.csv` in the sidebar to get started.")
    st.stop()

model, metrics, X, y, y_test, y_test_pred = train(df_raw)
means = X.mean()

# ----------------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📝 Student Profile")
    st.caption("The prediction updates live as you change values.")

    st.markdown("**📚 Academics**")
    hours = st.slider("Hours studied / week", 0, 44, 20)
    attendance = st.slider("Attendance (%)", 60, 100, 85)
    previous = st.slider("Previous score", 50, 100, 75)
    tutoring = st.slider("Tutoring sessions / month", 0, 8, 1)

    st.markdown("**🌱 Lifestyle**")
    sleep = st.slider("Sleep hours / night", 4, 10, 7)
    activity = st.slider("Physical activity (hrs / week)", 0, 6, 3)
    motivation = st.select_slider("Motivation level", ["Low", "Medium", "High"], "Medium")
    peer = st.select_slider("Peer influence", ["Negative", "Neutral", "Positive"], "Neutral")

    st.markdown("**🏠 Background**")
    parental = st.select_slider("Parental involvement", ["Low", "Medium", "High"], "Medium")
    resources = st.select_slider("Access to resources", ["Low", "Medium", "High"], "Medium")
    income = st.select_slider("Family income", ["Low", "Medium", "High"], "Medium")
    c1, c2 = st.columns(2)
    gender = c1.selectbox("Gender", ["Male", "Female"])
    internet = c2.selectbox("Internet", ["Yes", "No"])
    disability = st.selectbox("Learning disability", ["No", "Yes"])


# Build the input row in the exact feature order used for training
row = pd.DataFrame([{
    "Hours_Studied": hours, "Attendance": attendance,
    "Parental_Involvement": MAPPINGS["Parental_Involvement"][parental],
    "Gender": MAPPINGS["Gender"][gender],
    "Access_to_Resources": MAPPINGS["Access_to_Resources"][resources],
    "Tutoring_Sessions": tutoring, "Sleep_Hours": sleep, "Previous_Scores": previous,
    "Motivation_Level": MAPPINGS["Motivation_Level"][motivation],
    "Internet_Access": MAPPINGS["Internet_Access"][internet],
    "Family_Income": MAPPINGS["Family_Income"][income],
    "Peer_Influence": MAPPINGS["Peer_Influence"][peer],
    "Physical_Activity": activity,
    "Learning_Disabilities": MAPPINGS["Learning_Disabilities"][disability],
}])[FEATURES]

# Predict live from the current inputs
score = float(np.clip(model.predict(row)[0], 0, 100))
shown_row = row
letter, label, color = grade_for(score)
avg_score = float(y.mean())

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "🔍 Insights", "📊 Model Performance"])

# ---------------- Tab 1: Prediction ----------------
with tab1:
    left, right = st.columns([1.15, 1])

    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            number={"suffix": " / 100", "font": {"size": 44}},
            delta={"reference": avg_score, "increasing": {"color": "#16a34a"},
                   "decreasing": {"color": "#dc2626"}, "suffix": " vs avg"},
            title={"text": "Predicted Exam Score", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 50], "color": "rgba(220,38,38,.18)"},
                    {"range": [50, 60], "color": "rgba(234,88,12,.18)"},
                    {"range": [60, 70], "color": "rgba(217,119,6,.18)"},
                    {"range": [70, 80], "color": "rgba(37,99,235,.18)"},
                    {"range": [80, 100], "color": "rgba(22,163,74,.18)"},
                ],
                "threshold": {"line": {"color": "#7c3aed", "width": 4},
                              "thickness": .8, "value": avg_score},
            },
        ))
        fig.update_layout(height=340, margin=dict(t=70, b=10, l=30, r=30),
                          paper_bgcolor="rgba(0,0,0,0)", font={"family": "Poppins"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Purple line = dataset average ({avg_score:.1f})")

    with right:
        st.markdown(
            f"""
            <div class="card" style="text-align:center;">
                <div class="section-title">Result</div>
                <div style="font-size:4.5rem;font-weight:700;color:{color};line-height:1;">{letter}</div>
                <div style="margin:.6rem 0 1rem 0;">
                    <span class="grade-pill" style="background:{color};">{label}</span>
                </div>
                <div style="opacity:.8;">Predicted score <b>{score:.1f}</b> out of 100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💡 Personalised tips")
    r = shown_row.iloc[0]
    tips = []
    if r.Hours_Studied < means.Hours_Studied:
        tips.append(("warn", f"Studying **{r.Hours_Studied:.0f} h/week** is below average ({means.Hours_Studied:.0f} h). More focused study time is the strongest lever."))
    else:
        tips.append(("good", "Study hours are at or above average — keep the routine going."))
    if r.Attendance < 80:
        tips.append(("warn", f"Attendance of **{r.Attendance:.0f}%** is low. Aim for 85%+."))
    else:
        tips.append(("good", "Great attendance record."))
    if r.Sleep_Hours < 6:
        tips.append(("warn", "Sleeping under 6 hours can hurt focus and memory. Aim for 7–8 hours."))
    if r.Tutoring_Sessions == 0:
        tips.append(("warn", "No tutoring sessions — even 1–2 a month can help in weaker subjects."))
    if r.Motivation_Level == 0:
        tips.append(("warn", "Low motivation — set small, achievable weekly goals to rebuild momentum."))
    if r.Access_to_Resources == 0:
        tips.append(("warn", "Limited resources — school library, free online material and teachers can fill the gap."))
    if r.Peer_Influence == 0:
        tips.append(("warn", "Negative peer influence — studying with motivated classmates can shift this."))
    if r.Parental_Involvement == 2:
        tips.append(("good", "Strong parental involvement is a positive factor."))
    for kind, text in tips:
        st.markdown(f'<div class="tip {kind}">{text}</div>', unsafe_allow_html=True)

# ---------------- Tab 2: Insights ----------------
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### ⚖️ What's driving this score?")
        st.caption("Each bar shows how much a feature pushes this student's score above or below an average student.")
        contrib = pd.Series(model.coef_ * (shown_row.iloc[0].values - means.values), index=FEATURES)
        contrib = contrib.rename(NICE_NAMES).sort_values()
        top = pd.concat([contrib.head(5), contrib.tail(5)]).drop_duplicates()
        fig = go.Figure(go.Bar(
            x=top.values, y=top.index, orientation="h",
            marker_color=["#dc2626" if v < 0 else "#16a34a" for v in top.values],
            text=[f"{v:+.2f}" for v in top.values], textposition="outside",
        ))
        fig.update_layout(height=420, margin=dict(l=10, r=40, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Impact on score (points)", font={"family": "Poppins"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### 🧪 What-if: change one thing")
        st.caption("See how the prediction moves if a single habit changes.")
        lever = st.selectbox("Feature to change", ["Hours_Studied", "Attendance", "Sleep_Hours",
                                                    "Tutoring_Sessions", "Previous_Scores", "Physical_Activity"],
                             format_func=lambda k: NICE_NAMES[k])
        lo, hi = int(X[lever].min()), int(X[lever].max())
        grid = np.linspace(lo, hi, 25)
        sim = pd.concat([shown_row] * len(grid), ignore_index=True)
        sim[lever] = grid
        preds = np.clip(model.predict(sim[FEATURES]), 0, 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=preds, mode="lines", line=dict(color="#7c3aed", width=4),
                                 fill="tozeroy", fillcolor="rgba(124,58,237,.12)", name="Predicted"))
        fig.add_trace(go.Scatter(x=[shown_row.iloc[0][lever]], y=[score], mode="markers",
                                 marker=dict(size=14, color="#db2777", line=dict(width=2, color="white")),
                                 name="Current"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_title=NICE_NAMES[lever], yaxis_title="Predicted score",
                          yaxis=dict(range=[max(0, preds.min() - 5), min(100, preds.max() + 5)]),
                          font={"family": "Poppins"}, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        delta = preds[-1] - preds[0]
        st.info(f"Moving **{NICE_NAMES[lever]}** from {lo} to {hi} changes the prediction by about **{delta:+.1f} points**.")

    st.markdown("### 🏆 Overall feature importance")
    st.caption("Linear-regression coefficient × feature spread (std dev) — comparable across features.")
    imp = pd.Series(np.abs(model.coef_) * X.std().values, index=FEATURES).rename(NICE_NAMES).sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker=dict(color=imp.values, colorscale="Purples")))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"family": "Poppins"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab 3: Model performance ----------------
with tab3:
    st.markdown("### 📈 Test-set performance (Linear Regression)")
    cols = st.columns(5)
    items = [("R² Score", f"{metrics['R2']:.3f}"), ("MAE", f"{metrics['MAE']:.2f}"),
             ("MSE", f"{metrics['MSE']:.2f}"), ("RMSE", f"{metrics['RMSE']:.2f}"),
             ("Train R²", f"{metrics['train_r2']:.3f}")]
    for col, (lbl, val) in zip(cols, items):
        col.markdown(f'<div class="metric-box"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                     unsafe_allow_html=True)

    gap = metrics["train_r2"] - metrics["R2"]
    st.write("")
    if gap < 0.05:
        st.success(f"Test R² is as good as or better than train R² (train − test = {gap:+.3f}) — no sign of overfitting.")
    else:
        st.warning(f"Train and test R² differ by {gap:+.3f} — possible overfitting.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Actual vs Predicted")
        lim = [float(min(y_test.min(), y_test_pred.min())), float(max(y_test.max(), y_test_pred.max()))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_test_pred, mode="markers",
                                 marker=dict(color="#7c3aed", opacity=.55, size=6), name="Students"))
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                 line=dict(color="#dc2626", dash="dash"), name="Perfect"))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Actual", yaxis_title="Predicted",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"family": "Poppins"}, legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Residual distribution")
        resid = y_test.values - y_test_pred
        fig = go.Figure(go.Histogram(x=resid, nbinsx=40, marker_color="#db2777", opacity=.8))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Error (actual − predicted)", yaxis_title="Count",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"family": "Poppins"})
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📂 Peek at the dataset"):
        st.dataframe(df_raw.head(50), use_container_width=True)
        st.caption(f"{len(df_raw):,} rows × {df_raw.shape[1]} columns")

st.markdown(
    "<p style='text-align:center;opacity:.55;margin-top:2rem;'>Built with Streamlit · scikit-learn · Plotly</p>",
    unsafe_allow_html=True,
)