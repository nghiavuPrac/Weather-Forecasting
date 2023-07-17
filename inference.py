# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import GRU, Dense
# from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from category_encoders import BinaryEncoder
import pandas as pd

import numpy as np

def get_n_latest(df, Province, number_of_days):
  province_data = df[df['province']==Province][-number_of_days:].drop('province', axis=1).values.tolist().copy()

  lastest = pd.DataFrame(province_data,columns=['max','min','wind','wind_d','rain','humidi','cloud','pressure','date'])
  # lastest = lastest.set_index("date")
  # print(lastest['date'].iloc[-1] + pd.Timedelta(days=1))
  return lastest
  

def predicting(df, Province, predicted_number_of_days, model):

  encoded_df = df.copy()

  # Binary encode province feature
  encoded_df['copy_province'] = encoded_df['province']
  province_encoder = BinaryEncoder(cols=['copy_province'])
  encoded_df = province_encoder.fit_transform(encoded_df)

  binary_province_feature = ['copy_province_0', 'copy_province_1', 'copy_province_2', 'copy_province_3', 'copy_province_4', 'copy_province_5']
  encoded_df[binary_province_feature] = encoded_df[binary_province_feature].astype('int8')

  def cap_and_floor_outliers_iqr(df, columns, k=1.5):
      # Capping and flooring outliers
      in_outlier=list()
      for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        count= len(df[(df[column] > lower_bound) & (df[column]< upper_bound)])
        in_outlier.append(count)

        df[column] = df[column].clip(lower_bound, upper_bound)

      return df, in_outlier

  encoded_df, _ = cap_and_floor_outliers_iqr(encoded_df, ['max', 'min', 'pressure', 'humidi', 'cloud', 'wind'])

  # Select feature

  provinces = ['province', 'copy_province_0', 'copy_province_1','copy_province_2', 'copy_province_3', 'copy_province_4','copy_province_5']
  normalize_feature = ['max', 'min', 'rain', 'pressure', 'humidi', 'cloud', 'wind']
  num_feature = len(normalize_feature)


  # Min max scale
  scaled_columns = ['_'.join(['scaled', column]) for column in normalize_feature]

  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(encoded_df[normalize_feature])
  encoded_df[scaled_columns] = pd.DataFrame(scaled_data, columns=scaled_columns)

  selected_df = encoded_df[provinces+scaled_columns]


  # Parameters
  past = 60 # Number of days in the past used to predict
  future_length = 5


  #@title Input fields
  dayth = 0

  # Province = "Bac Lieu" #@param ["Bac Lieu", "Ben Tre", "Bien Hoa", "Buon Me Thuot", "Ca Mau", "Cam Pha", "Cam Ranh", "Can Tho", "Chau Doc", "Da Lat", "Ha Noi", "Hai Duong", "Hai Phong", "Hanoi", "Ho Chi Minh City", "Hoa Binh", "Hong Gai", "Hue", "Long Xuyen", "My Tho", "Nam Dinh", "Nha Trang", "Phan Rang", "Phan Thiet", "Play Cu", "Qui Nhon", "Rach Gia", "Soc Trang", "Tam Ky", "Tan An", "Thai Nguyen", "Thanh Hoa", "Tra Vinh", "Tuy Hoa", "Uong Bi", "Viet Tri", "Vinh", "Vinh Long", "Vung Tau", "Yen Bai"]
  # predicted_number_of_days = 5 #@param {type:"integer"}

  # Get input data
  input = selected_df[selected_df['province']==Province][-past:].drop('province', axis=1).values.tolist().copy()

  province_encoder = BinaryEncoder(cols=['province'])
  province_encoder.fit_transform(df['province'])
  encoded_province = province_encoder.transform(pd.DataFrame({'province': [Province]})).values.tolist()[0]

  # Predict
  for day in range(predicted_number_of_days):
    input.append(encoded_province + model.predict([input[day:(day+past)]])[0].reshape((future_length, num_feature))[dayth].tolist())

  result = pd.DataFrame(np.array(input), columns=selected_df.columns[1:])

  past_date = df['date'][-past:]
  predict_date = [df['date'].iloc[-1] + pd.Timedelta(days=i) for i in range(predicted_number_of_days+1)]


  rescale_data = np.round(scaler.inverse_transform(result.values[:, -7:]), 2)
  predict_df = pd.DataFrame(rescale_data, columns=normalize_feature)
  predict_df.index = list(past_date[:-1])+predict_date
  # print(predict_df[-predicted_number_of_days:])
  return predict_df



if __name__ == '__main__':
  df_path = './DBM301/clean_data.csv'
  df = pd.read_csv(df_path, index_col=[0])
  df['date']=pd.to_datetime(df['date'])
  
  Province = "Bac Lieu" #@param ["Bac Lieu", "Ben Tre", "Bien Hoa", "Buon Me Thuot", "Ca Mau", "Cam Pha", "Cam Ranh", "Can Tho", "Chau Doc", "Da Lat", "Ha Noi", "Hai Duong", "Hai Phong", "Hanoi", "Ho Chi Minh City", "Hoa Binh", "Hong Gai", "Hue", "Long Xuyen", "My Tho", "Nam Dinh", "Nha Trang", "Phan Rang", "Phan Thiet", "Play Cu", "Qui Nhon", "Rach Gia", "Soc Trang", "Tam Ky", "Tan An", "Thai Nguyen", "Thanh Hoa", "Tra Vinh", "Tuy Hoa", "Uong Bi", "Viet Tri", "Vinh", "Vinh Long", "Vung Tau", "Yen Bai"]
  predicted_number_of_days = 5 #@param {type:"integer"}

  lastest= get_n_latest(df,Province, predicted_number_of_days)


  # predict=predicting(df,Province, predicted_number_of_days)
  # print(predict)