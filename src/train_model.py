import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------
# 1. Load dataset
# --------------------------------

data = pd.read_csv("data/payasam_data.csv")

print("Dataset loaded successfully!")
print("Number of samples:", len(data))


# --------------------------------
# 2. Separate input and output
# --------------------------------

X = data.drop("consistency_score", axis=1)
y = data["consistency_score"]


# --------------------------------
# 3. Identify columns
# --------------------------------

categorical_features = ["payasam_type"]

numerical_features = [
    "milk_ml",
    "water_ml",
    "main_ingredient_g",
    "sugar_g",
    "cooking_time_min",
    "temperature_c"
]


# --------------------------------
# 4. Preprocessing
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# --------------------------------
# 5. Create Random Forest model
# --------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=None
)


# --------------------------------
# 6. Create complete ML pipeline
# --------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------
# 7. Split dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------
# 8. Train model
# --------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# --------------------------------
# 9. Make predictions
# --------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------
# 10. Evaluate model
# --------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


print("\nModel Performance")
print("-------------------------")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))


# --------------------------------
# 11. Save trained model
# --------------------------------

joblib.dump(
    pipeline,
    "models/payasam_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/payasam_model.pkl")