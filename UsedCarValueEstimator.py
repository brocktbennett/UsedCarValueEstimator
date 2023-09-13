import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data():
    """Load the used car data set."""
    column_names = [
        "index",
        "dateCrawled",
        "name",
        "seller",
        "offerType",
        "price",
        "abtest",
        "vehicleType",
        "yearOfRegistration",
        "gearbox",
        "powerPS",
        "model",
        "kilometer",
        "monthOfRegistration",
        "brand",
        "notRepairedDamage",
        "dateCreated",
        "nrOfPictures",
        "postalCode",
        "lastSeen",
    ]
    df_auto = pd.read_csv("/Users/brocktbennett/GitHub/UsedCarValueEstimator/autos.csv", header=0, names=column_names)

    return df_auto


def clean_data(df_auto):
    """Clean the used car data set."""

    df_auto.drop(
        ["index", "dateCrawled", "name", "dateCreated", "lastSeen"], axis=1, inplace=True
    )

    cat_columns = [
        "seller",
        "offerType",
        "abtest",
        "vehicleType",
        "gearbox",
        "model",
        "brand",
        "notRepairedDamage",
        "fuelType",
    ]
    df_auto = pd.get_dummies(df_auto, columns=cat_columns)
    df_auto["fuelType"] = df_auto["fuelType"].fillna("missing")

    df_auto["car_age"] = 2023 - df_auto["yearOfRegistration"]

    return df_auto


def train_and_evaluate_models(df_auto):
    """Train and evaluate the linear regression and random forest models."""

    X = df_auto.drop("price", axis=1)
    y = df_auto["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin_reg = lin_reg.predict(X_test)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lin_reg)))
    print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))


def main():
    """Load the data, clean it, train and evaluate the models."""
    df_auto = load_data()
    df_auto = clean_data(df_auto)
    train_and_evaluate_models(df_auto)


if __name__ == "__main__":
    main()
