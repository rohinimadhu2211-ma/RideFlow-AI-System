from fastapi import FastAPI

import pandas as pd

import joblib


# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(
    "ride_demand_prediction_model.pkl"
)


# =====================================================
# CREATE FASTAPI APP
# =====================================================

app = FastAPI()


# =====================================================
# HOME API
# =====================================================

@app.get("/")

def home():

    return {

        "message":

        "RideFlow AI FastAPI Running"
    }


# =====================================================
# DEMAND PREDICTION API
# =====================================================

@app.get("/predict_demand")

def predict_demand(

    pickup_zone: int,

    hour: int,

    weekday: int
):


    # LOAD DATASET

    df = pd.read_csv(
        "rideflow_feature_engineered.csv"
    )


    # TAKE SAMPLE ROW

    sample = df.iloc[[0]].copy()


    # MODIFY USER INPUTS

    sample["pickup_zone"] = pickup_zone

    sample["hour"] = hour

    sample["weekday"] = weekday


    # MATCH MODEL FEATURES

    required_features = model.feature_names_in_

    sample = sample[required_features]


    # MODEL PREDICTION

    prediction = model.predict(
        sample
    )[0]


    # OUTPUT RESULT

    if prediction == 1:

        result = "High Demand"

    else:

        result = "Low Demand"


    return {

        "prediction": result
    }