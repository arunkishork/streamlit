import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Bank Churn Dashboard", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}
h1 {
    text-align: center;
    color: #1f3b4d;
}
.metric-box {
    background-color: white;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Bank Customer Churn Analysis Dashboard")

# ---------------- LOAD ----------------
model = pickle.load(open('model.pkl', 'rb'))
data = pd.read_csv("Customer-Churn-Records.csv")

# ---------------- LAYOUT ----------------
left, right = st.columns([1, 3])   # LEFT SMALL, RIGHT BIG

# =========================================================
# 🤖 LEFT SIDE → PREDICTION (SMALL PANEL)
# =========================================================
with left:
    st.subheader("🔮 Predict Churn")

    age = st.number_input("Age", 18, 100, 40)
    credit = st.number_input("Credit Score", 300, 900, 650)
    balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
    geo = st.selectbox("Geography", ["France","Germany","Spain"])
    active = st.selectbox("Active Member", [0,1])

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            'CreditScore': credit,
            'Age': age,
            'Balance': balance,
            'Geography': geo,
            'IsActiveMember': active
        }])

        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"⚠️ Churn Risk: {round(prob*100,2)}%")
        else:
            st.success(f"✅ Safe Customer: {round((1-prob)*100,2)}%")

# =========================================================
# 📊 RIGHT SIDE → FULL DASHBOARD
# =========================================================
with right:

    # ---------- KPIs ----------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", len(data))
    k2.metric("Churn %", round(data['Exited'].mean()*100,2))
    k3.metric("Avg Balance", int(data['Balance'].mean()))
    k4.metric("Avg Age", int(data['Age'].mean()))

    st.divider()

    # ---------- ROW 1 ----------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Exited', data=data, palette='coolwarm', ax=ax)
        st.pyplot(fig)

    with c2:
        st.subheader("Gender vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(x='Gender', hue='Exited', data=data, palette='Set2', ax=ax)
        st.pyplot(fig)

    with c3:
        st.subheader("Geography vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(x='Geography', hue='Exited', data=data, palette='Set1', ax=ax)
        st.pyplot(fig)

    # ---------- ROW 2 ----------
    c4, c5, c6 = st.columns(3)

    with c4:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data=data, x='Age', hue='Exited', bins=25, ax=ax)
        st.pyplot(fig)

    with c5:
        st.subheader("Balance vs Churn")
        fig, ax = plt.subplots()
        sns.boxplot(x='Exited', y='Balance', data=data, ax=ax)
        st.pyplot(fig)

    with c6:
        st.subheader("Products vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(x='NumOfProducts', hue='Exited', data=data, ax=ax)
        st.pyplot(fig)

