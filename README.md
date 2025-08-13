# credit-risk-prediction
.Credit Risk Prediction Model predicts loan repayment probability using EDA, feature engineering, Random Forest, LightGBM, ROC-AUC evaluation, and explainability analysis.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# VERİYİ YÜKLE (Sütun isimleriyle birlikte)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = [
    'existing_checking', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings', 'employment', 'installment_rate', 'personal_status', 'debtors',
    'residence', 'property', 'age', 'other_plans', 'housing', 'existing_credits',
    'job', 'dependents', 'telephone', 'foreign_worker', 'risk'  # Buradaki isim 'risk' olarak tanımlandı!
]
df = pd.read_csv(url, delim_whitespace=True, header=None, names=columns)

# EKSİK VERİ KONTROLÜ
print("Eksik veri kontrolü:\n", df.isnull().sum())

# TARGET DAĞILIMI (Artık 'risk' sütunu var)
df['risk'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title("Risk Dağılımı (1: Good, 2: Bad)")
plt.show()

# KORELASYON MATRİSİ
sns.heatmap(df.corr(), annot=True)
plt.show()

# credit_analysis.py
import pandas as pd
import numpy as np

# 1. VERİ YÜKLEME (DÜZGÜN YAZIM)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ['existing_checking', 'duration', 'credit_history', 'purpose', 'credit_amount',
           'savings', 'employment', 'installment_rate', 'personal_status', 'debtors',
           'residence', 'property', 'age', 'other_plans', 'housing', 'existing_credits',
           'job', 'dependents', 'telephone', 'foreign_worker', 'risk']
df = pd.read_csv(url, delim_whitespace=True, header=None, names=columns)  # DİKKAT: delim_whitespace

# 2. ÖZELLİK MÜHENDİSLİĞİ
# Gelir sütunu ekle (örnek veri)
np.random.seed(42)
df['income'] = np.random.randint(1000, 5000, size=len(df))

# Yeni özellikler
df['debt_to_income_ratio'] = df['credit_amount'] / df['income']
df['age_group'] = pd.cut(df['age'], bins=[18, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])

# 3. ÇIKTI KONTROLÜ
print("\nYENİ SÜTUNLAR:")
print(df[['debt_to_income_ratio', 'age_group', 'income']].head())  # Yeni sütunları göster
print("\nVeri boyutu:", df.shape)

# random_forest_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Veri yükleme
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = [
    'existing_checking', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings', 'employment', 'installment_rate', 'personal_status', 'debtors',
    'residence', 'property', 'age', 'other_plans', 'housing', 'existing_credits',
    'job', 'dependents', 'telephone', 'foreign_worker', 'risk'
]
df = pd.read_csv(url, delim_whitespace=True, header=None, names=columns)

# Özellik mühendisliği
df['risk'] = df['risk'].map({1: 0, 2: 1})  # Riskli=1, Güvenli=0
X = pd.get_dummies(df.drop('risk', axis=1))
y = df['risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Sonuçlar
print("📊 Model Performansı:\n")
print(classification_report(y_test, model.predict(X_test)))


import pandas as pd
from urllib.request import urlretrieve

try:
    # 1. DOSYAYI İNTERNETTEN ÇEK
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    urlretrieve(url, "german.data")  # Bulunduğunuz klasöre indirir
    
    # 2. OKU
    df = pd.read_csv("german.data", sep='\\s+', header=None, encoding='latin-1')
    print("✅ İNTERNETTEN OKUNDU! İlk 3 satır:")
    print(df.head(3))

except Exception as e:
    print(f"❌ KRİTİK HATA: {str(e)}")
    
