
import numpy as np
import pandas as pd 
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
d=pd.read_csv("data.csv")
x_train = d[[
    "Fan", "Fan Power (W)", "Fan Hours/Day",
    "LED Bulb", "LED Bulb Power (W)", "LED Bulb Hours/Day",
    "Refrigerator", "Refrigerator Power (W)", "Refrigerator Hours/Day",
    "AC", "AC Power (W)", "AC Hours/Day",
    "TV", "TV Power (W)", "TV Hours/Day",
    "Rice Cooker", "Rice Cooker Power (W)", "Rice Cooker Hours/Day",
    "Electric Kettle", "Electric Kettle Power (W)", "Electric Kettle Hours/Day",
    "Air Fryer", "Air Fryer Power (W)", "Air Fryer Hours/Day",
    "Oven", "Oven Power (W)", "Oven Hours/Day",
    "Iron", "Iron Power (W)", "Iron Hours/Day",
    "Washing Machine", "Washing Machine Power (W)", "Washing Machine Hours/Day",
    "Water Pump", "Water Pump Power (W)", "Water Pump Hours/Day",
    "WiFi Router", "WiFi Router Power (W)", "WiFi Router Hours/Day",
    "Blender", "Blender Power (W)", "Blender Hours/Day"
    
]].to_numpy()
y_train = np.array(d["Monthly Energy (kWh)"])
# feature scaling
scale=StandardScaler()
x_train_normalized=scale.fit_transform(x_train)
#linear regression
regression=SGDRegressor(max_iter=1000000)
regression.fit(x_train_normalized,y_train)
w_normalized=regression.coef_
b_normalized=regression.intercept_
# input 
appliances = [
    "Fan", "LED Bulb", "Refrigerator", "AC", "TV",
    "Rice Cooker", "Electric Kettle", "Air Fryer", "Oven",
    "Iron", "Washing Machine", "Water Pump", "WiFi Router", "Blender"
]
user_input = []
for appliance in appliances:
    number = float(input(f"Enter number of {appliance}: "))
    power = float(input(f"Enter power of one {appliance} (W): "))
    hours = float(input(f"Enter hours used per day for {appliance}: "))
    user_input.extend([number, power, hours])
x_new = np.array([user_input])
x_new_normalized = scale.transform(x_new)
prediction = regression.predict(x_new_normalized)
print(f"\nPredicted monthly energy consumption: {prediction[0]:.2f} kWh")
 
