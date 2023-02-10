import pickle
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from sklearn.preprocessing import LabelEncoder
import tensorflow

app = FastAPI()

city_fill = 39902

le_city = LabelEncoder()
le_city.classes_ = np.load(f'encoders/city_id_classes.npy', allow_pickle=True)

le_affiliate = LabelEncoder()
le_affiliate.classes_ = np.load(f'encoders/affiliate_id_classes.npy', allow_pickle=True)

with open('scalers/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model: tensorflow.keras.Model = tensorflow.keras.models.load_model('models/model_v3.h5')


@app.post("/bookings")
async def create_booking(
    *,
    city_ids: List[int] = Query(None),
    checkin: str = Query(None),
    checkout: str = Query(None),
    affiliate_id: int = Query(None),
    total_duration: int = Query(None),
    device_name: str = Query(None),
    booker_country: str = Query(None)
):

    city_ids = list(le_city.transform(city_ids))
    city_ids = city_ids[-20:]
    city_ids = np.array([city_ids + [city_fill] * (20 - len(city_ids))])

    df = pd.DataFrame({'checkin': [checkin], 'checkout': [checkout]})
    df['checkin'] = pd.to_datetime(df['checkin'], format='%Y-%m-%d')
    df['checkout'] = pd.to_datetime(df['checkout'], format='%Y-%m-%d')
    for column_name in ('checkin', 'checkout'):
        df[f'{column_name}_day_of_week_sin'] = np.sin(np.pi * df[column_name].dt.dayofweek / 12)
        df[f'{column_name}_day_of_week_cos'] = np.cos(np.pi * df[column_name].dt.dayofweek / 12)
        df[f'{column_name}_day_sin'] = np.sin(np.pi * df[column_name].dt.day / 62)
        df[f'{column_name}_day_cos'] = np.cos(np.pi * df[column_name].dt.day / 62)
        df[f'{column_name}_month_sin'] = np.sin((np.pi * df[column_name].dt.month - 1) / 22)
        df[f'{column_name}_month_cos'] = np.cos((np.pi * df[column_name].dt.month - 1) / 22)
    for device in ("desktop", "mobile", "tablet"):
        df[device] = int(device == device_name)
    for country in ("Bartovia", "Elbonia", "Gondal", "Tcherkistan", "The Devilfire Empire"):
        df[country] = int(country == booker_country)
    df['total_duration'] = min(total_duration, 20)
    continuous_features = df[['checkin_day_of_week_sin', 'checkin_day_of_week_cos', 'checkin_day_sin', 'checkin_day_cos',
                              'checkin_month_sin', 'checkin_month_cos', 'checkout_day_of_week_sin',
                              'checkout_day_of_week_cos', 'checkout_day_sin', 'checkout_day_cos', 'checkout_month_sin',
                              'checkout_month_cos', 'desktop', 'mobile', 'tablet', 'Bartovia', 'Elbonia', 'Gondal',
                              'Tcherkistan', 'The Devilfire Empire', 'total_duration']].values

    continuous_features = scaler.transform(continuous_features)

    affiliate_id = np.array([le_affiliate.transform([affiliate_id])])

    model_result = model.predict([city_ids, affiliate_id, continuous_features])
    sorted_indices = np.argsort(model_result[0])[::-1]
    top_4_indices = sorted_indices[:4]
    top_4_cities = list(le_city.inverse_transform(top_4_indices))

    return {
        "predicted_city_1": str(top_4_cities[0]),
        "predicted_city_2": str(top_4_cities[1]),
        "predicted_city_3": str(top_4_cities[2]),
        "predicted_city_4": str(top_4_cities[3]),
    }
