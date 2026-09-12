import pandas as pd
import joblib


# Load trained model
model = joblib.load("models/payasam_model.pkl")

print("====================================")
print("   PIA - Payasam Intelligence Agency")
print("====================================")
print("       Consistency Predictor")
print()


# Get user input
payasam_type = input(
    "Enter payasam type (Palada/Semiya/Rice/Parippu/Paal): "
)

milk_ml = float(input("Enter milk quantity (ml): "))
water_ml = float(input("Enter water quantity (ml): "))
main_ingredient_g = float(
    input("Enter main ingredient quantity (g): ")
)
sugar_g = float(input("Enter sugar quantity (g): "))
cooking_time_min = float(
    input("Enter cooking time (minutes): ")
)
temperature_c = float(
    input("Enter temperature (°C): ")
)


# Create input DataFrame
new_payasam = pd.DataFrame({
    "payasam_type": [payasam_type],
    "milk_ml": [milk_ml],
    "water_ml": [water_ml],
    "main_ingredient_g": [main_ingredient_g],
    "sugar_g": [sugar_g],
    "cooking_time_min": [cooking_time_min],
    "temperature_c": [temperature_c]
})


# Predict
prediction = model.predict(new_payasam)[0]


# Keep score between 0 and 100
prediction = max(0, min(100, prediction))


print()
print("====================================")
print("          PIA RESULT")
print("====================================")
print(
    f"Predicted Consistency Score: {prediction:.2f} / 100"
)


# Give a simple interpretation
if prediction < 40:
    level = "Very Thin"
elif prediction < 55:
    level = "Thin"
elif prediction < 70:
    level = "Medium"
elif prediction < 85:
    level = "Thick"
else:
    level = "Very Thick"


print("Consistency Level:", level)
print("====================================")