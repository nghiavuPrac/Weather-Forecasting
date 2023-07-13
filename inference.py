# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import GRU, Dense
# from keras.callbacks import EarlyStopping
import keras
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from category_encoders import BinaryEncoder

import pandas as pd
import os
import numpy as np

path = './DBM301/encoded.csv'
encoded_df = pd.read_csv(path)

past = 60 # Number of days in the past used to predict
step = 1 # Distance between days
distin_feature = 'province'
# target = scaled_columns + ['province']
# df_length = len(selected_df)
future_length = 5
ratio = 0.8
dayth = 0


# Select feature
features = ['max', 'min', 'rain', 'pressure', 'humidi', 'cloud', 'wind', 'province', 'copy_province_0', 'copy_province_1',
       'copy_province_2', 'copy_province_3', 'copy_province_4', 'copy_province_5']
normalize_feature = ['max', 'min', 'rain', 'pressure', 'humidi', 'cloud', 'wind']
num_feature = len(normalize_feature)

selected_df = encoded_df[features]

# Min max scale

scaled_columns = ['_'.join(['scaled', column]) for column in normalize_feature]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(selected_df[normalize_feature])
selected_df[scaled_columns] = pd.DataFrame(scaled_data, columns=scaled_columns)

selected_df = selected_df.drop(normalize_feature, axis=1)

#load model
file_path = './DBM301/model2.h5'
overlap = True
history = []


if os.path.exists(file_path) and overlap==True:
    model = keras.models.load_model(file_path)
else:
    print("Loading Fail!")
    exit()


province_list=[]


#Input fields

Province = "Bac Lieu" #@param ["Bac Lieu", "Ben Tre", "Bien Hoa", "Buon Me Thuot", "Ca Mau", "Cam Pha", "Cam Ranh", "Can Tho", "Chau Doc", "Da Lat", "Ha Noi", "Hai Duong", "Hai Phong", "Hanoi", "Ho Chi Minh City", "Hoa Binh", "Hong Gai", "Hue", "Long Xuyen", "My Tho", "Nam Dinh", "Nha Trang", "Phan Rang", "Phan Thiet", "Play Cu", "Qui Nhon", "Rach Gia", "Soc Trang", "Tam Ky", "Tan An", "Thai Nguyen", "Thanh Hoa", "Tra Vinh", "Tuy Hoa", "Uong Bi", "Viet Tri", "Vinh", "Vinh Long", "Vung Tau", "Yen Bai"]
predicted_number_of_days = 2 #@param {type:"integer"}

# Get input data
input = selected_df[selected_df['province']==Province][-past:].drop('province', axis=1).values.tolist().copy()
index = encoded_df[selected_df['province']==Province][-past:]

province_encoder = BinaryEncoder(cols=['province'])
province_encoder.fit_transform(encoded_df['province'])
encoded_province = province_encoder.transform(pd.DataFrame({'province': [Province]})).values.tolist()[0]

# Predict
for day in range(predicted_number_of_days):
  input.append(encoded_province + model.predict([input[day:(day+past)]])[0].reshape((future_length, num_feature))[dayth].tolist())

result = pd.DataFrame(np.array(input), columns=selected_df.columns[1:])


# Rescale prediction
# s1 = scaler.inverse_transform(result[scaled_columns])
# result[normalize_feature] = pd.DataFrame(s1, columns=normalize_feature)
# result = result.drop(scaled_columns, axis=1)

past_date = encoded_df['date'][-past:]
predict_date = [encoded_df['date'].iloc[-1] + pd.Timedelta(days=i) for i in range(predicted_number_of_days+1)]

print(predict_date[0])
# print(type(past_date[0]))

rescale_data = np.round(scaler.inverse_transform(result.values[:, -7:]), 2)
predict_df = pd.DataFrame(rescale_data, columns=normalize_feature, index=predict_date)
predict_df[-5:]
print(predict_df)

# plt.figure(figsize=(22, 22))
# plt.subplots_adjust(wspace=0.4, hspace=0.6)

# for i in range(num_feature):
#   plt.subplot(num_feature, 1, i+1)
#   plt.plot(encoded_df['date'][-past:], result.iloc[:past][normalize_feature[i]], 'o-', label='Dữ liệu')
#   plt.plot(predict_date, result.iloc[-(predicted_number_of_days+1):][normalize_feature[i]], 'o-', color='orange', label='Dự đoán')
#   plt.title((normalize_feature[i]).upper())
#   plt.legend()


# plt.show()
