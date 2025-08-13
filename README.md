# credit-risk-prediction
.Credit Risk Prediction Model predicts loan repayment probability using EDA, feature engineering, Random Forest, LightGBM, ROC-AUC evaluation, and explainability analysis.
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# 1. Veriyi yükle
try:
    df = pd.read_csv('vehicle_sensor_data.csv')  # Dosya adını kontrol edin!
    print("Veri başarıyla yüklendi. İlk 5 satır:")
    print(df.head())
except FileNotFoundError:
    print("HATA: 'vehicle_sensor_data.csv' dosyası bulunamadı!")
    print("Lütfen dosya yolunu kontrol edin veya örnek bir veri seti oluşturun.")
    
    # Örnek veri oluştur (test için)
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='H'),
        'sensor1': [0.85, 0.87, 0.88, 0.82, 0.90],
        'sensor2': [1.2, 1.3, 1.1, 1.4, 1.0]
    }
    df = pd.DataFrame(data)
    print("\nÖrnek veri oluşturuldu:")
    print(df)

    print("\nEksik veri kontrolü:")
print(df.isnull().sum())

from sklearn.ensemble import IsolationForest

# Anomali tespiti
model = IsolationForest(contamination=0.1, random_state=42)
anomalies = model.fit_predict(df[['sensor1', 'sensor2']])
df['anomaly'] = anomalies  # -1: Anomali, 1: Normal

print("\nAnomali durumu:")
print(df['anomaly'].value_counts())

# Zaman damgası ekleme (örnek)
df['timestamp'] = pd.date_range(start='2023-01-01', periods=5, freq='H')

# Zaman serisi olarak ayarlama
df.set_index('timestamp', inplace=True)
print("\nZaman serisi verisi:")
print(df)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(df.index, df['sensor1'], label='Sensör 1', marker='o')
plt.plot(df.index, df['sensor2'], label='Sensör 2', marker='x')
plt.xlabel('Zaman')
plt.ylabel('Değer')
plt.title('Sensör Verileri Zaman Serisi')
plt.legend()
plt.grid()
plt.show()

df.to_csv('processed_sensor_data.csv')
print("\nVeri 'processed_sensor_data.csv' olarak kaydedildi!")

import pandas as pd
import numpy as np

# 1. ÖRNEK VERİ OLUŞTUR (Gerçek veriniz varsa bu adımı atlayın)
data = {
    'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='H'),
    'sensor1': [0.85, 0.87, 0.88, 0.82, 0.90],
    'sensor2': [1.20, 1.30, 1.10, 1.40, 1.00]
}
df = pd.DataFrame(data)
print("Örnek Veri:\n", df)

# 2. FEATURE ENGINEERING
window_size = 3  # 3 zaman birimlik pencere

for sensor in ['sensor1', 'sensor2']:
    # Temel istatistikler
    df[f'{sensor}_mean'] = df[sensor].rolling(window=window_size).mean()
    df[f'{sensor}_max'] = df[sensor].rolling(window=window_size).max()
    df[f'{sensor}_min'] = df[sensor].rolling(window=window_size).min()
    df[f'{sensor}_std'] = df[sensor].rolling(window=window_size).std()  # DİKKAT: "_std" olmalı (nokta değil)
    
    # Değişim hızı (% olarak)
    df[f'{sensor}_change_rate'] = df[sensor].pct_change() * 100
    
    # Mutlak değişim
    df[f'{sensor}_diff'] = df[sensor].diff()

# 3. SONUÇLARI GÖSTER
print("\nİşlenmiş Veri:\n", df)


import pandas as pd
import numpy as np

# 1. ÖRNEK VERİ OLUŞTUR (Gerçek veriniz varsa bu adımı atlayın)
data = {
    'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='H'),
    'sensor1': [0.85, 0.87, 0.88, 0.82, 0.90],
    'sensor2': [1.20, 1.30, 1.10, 1.40, 1.00]
}
df = pd.DataFrame(data)
print("Örnek Veri:\n", df)

# 2. FEATURE ENGINEERING
window_size = 3  # 3 zaman birimlik pencere

for sensor in ['sensor1', 'sensor2']:
    # Temel istatistikler
    df[f'{sensor}_mean'] = df[sensor].rolling(window=window_size).mean()
    df[f'{sensor}_max'] = df[sensor].rolling(window=window_size).max()
    df[f'{sensor}_min'] = df[sensor].rolling(window=window_size).min()
    df[f'{sensor}_std'] = df[sensor].rolling(window=window_size).std()  # DİKKAT: "_std" olmalı (nokta değil)
    
    # Değişim hızı (% olarak)
    df[f'{sensor}_change_rate'] = df[sensor].pct_change() * 100
    
    # Mutlak değişim
    df[f'{sensor}_diff'] = df[sensor].diff()

# 3. SONUÇLARI GÖSTER
print("\nİşlenmiş Veri:\n", df)

import numpy as np

# 1. Örnek arıza bayrağı oluştur (sensor1 > 0.88 ise arıza kabul edelim)
df['failure_flag'] = np.where(df['sensor1'] > 0.88, 1, 0)

# 2. TTF hesapla (arıza öncesi kalan adım sayısı)
df['TTF'] = df[::-1]['failure_flag'].cumsum()[::-1]  # Ters çevirip cumulative sum al

print(df[['timestamp', 'sensor1', 'failure_flag', 'TTF']])
