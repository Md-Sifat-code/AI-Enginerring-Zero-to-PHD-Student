import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# 1) Load
df = pd.read_csv(r"2nd Linear Regression Project\plastic_cashback_synthetic.csv", parse_dates=["date"])

# 2) Select features/target
features = [
    "cashback_bdt","bottle_size_ml","plastic_grade",
    "market_recycle_price_bdt_per_kg","user_density_per_km2",
    "awareness_index","distance_to_collection_km","rain_mm",
    "campaign","region"
]
X = df[features]
y = df["return_rate_percent"]

num_feats = [
    "cashback_bdt","bottle_size_ml","plastic_grade",
    "market_recycle_price_bdt_per_kg","user_density_per_km2",
    "awareness_index","distance_to_collection_km","rain_mm","campaign"
]
cat_feats = ["region"]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
])

# 3) Model + pipeline
pipe = Pipeline([
    ("pre", pre),
    ("model", Ridge())
])

# 4) Hyperparameters (simple)
param_grid = {"model__alpha": [0.1, 0.3, 1.0, 3.0, 10.0]}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gs = GridSearchCV(pipe, param_grid, cv=5, scoring="r2", n_jobs=-1)
gs.fit(X_train, y_train)

print("Best alpha:", gs.best_params_["model__alpha"])

# 5) Evaluate
pred = gs.predict(X_test)
print("R^2:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))

# 6) Persist
joblib.dump(gs.best_estimator_, "cashback_lr_pipeline.pkl")
print("Saved: cashback_lr_pipeline.pkl")
