
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Electricity Bill Predictor",
    page_icon="⚡"
)

st.title("⚡ Smart Electricity Bill Predictor")
st.write("Estimate your monthly household electricity consumption in Bangladesh.")

appliances = [
    "Fan", "LED Bulb", "Refrigerator", "AC", "TV",
    "Rice Cooker", "Electric Kettle", "Air Fryer", "Oven",
    "Iron", "Washing Machine", "Water Pump", "WiFi Router", "Blender"
]

features = []
for a in appliances:
    features.extend([a, f"{a} Power (W)", f"{a} Hours/Day"])

@st.cache_resource
def train_model():
    d = pd.read_csv("data.csv")
    X = d[features].to_numpy(dtype=float)
    y = d["Monthly Energy (kWh)"].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SGDRegressor(max_iter=100000, random_state=42)
    model.fit(X_scaled, y)

    return scaler, model

try:
    scaler, model = train_model()
except Exception as e:
    st.error(f"Could not load the dataset or train the model: {e}")
    st.stop()

st.subheader("Enter your appliances")
user_input = []

with st.form("appliances"):
    for a in appliances:
        st.markdown(f"**{a}**")
        c1, c2, c3 = st.columns(3)

        with c1:
            number = st.number_input(
                "Quantity", min_value=0, max_value=100,
                value=0, key=f"{a}_n"
            )
        with c2:
            power = st.number_input(
                "Power (W)", min_value=0.0, value=0.0,
                key=f"{a}_p"
            )
        with c3:
            hours = st.number_input(
                "Hours/day", min_value=0.0, max_value=24.0,
                value=0.0, key=f"{a}_h"
            )

        user_input.extend([number, power, hours])

    submitted = st.form_submit_button("Predict Monthly Bill")

def kwh_to_bill(kwh):
    remaining = max(0.0, kwh)
    cost = min(remaining, 75) * 5.26
    remaining -= min(remaining, 75)

    cost += min(remaining, 125) * 7.20
    return (cost + 42) * 1.05

if submitted:
    X_new = np.array([user_input], dtype=float)
    prediction = float(model.predict(scaler.transform(X_new))[0])

    if prediction < 0:
        st.warning("The model predicted negative consumption. Check the model and dataset.")
    else:
        bill = kwh_to_bill(prediction)

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Monthly Consumption", f"{prediction:.2f} kWh")
        c2.metric("Estimated Bill", f"৳{bill:.2f}")

        st.caption(
            "This is an estimate, not an official electricity bill. "
            "Tariff rates and charges should be verified before relying on the estimate."
        )

        if prediction > 200:
            st.warning(
                "The current bill calculation covers energy charges only "
                "up to 200 kWh. The bill may be underestimated."
            )
