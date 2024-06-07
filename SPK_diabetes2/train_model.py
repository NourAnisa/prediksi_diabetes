import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Membuat dataset dengan semua fitur yang diperlukan
data = {
    'frekuensi_buang_air_kecil': [1, 1, 0, 0, 1, 0],
    'rasa_haus': [1, 1, 0, 0, 1, 0],
    'kulit_mulut_kering': [1, 0, 0, 0, 1, 0],
    'penurunan_berat_badan': [1, 1, 0, 1, 0, 0],
    'kelelahan': [1, 1, 1, 0, 1, 0],
    'penglihatan_buram': [1, 0, 1, 0, 0, 0],
    'gatal_alat_kelamin': [0, 0, 1, 0, 1, 0],
    'penyembuhan_luka_lambat': [0, 0, 1, 0, 1, 0],
    'mata_kering': [0, 0, 1, 0, 1, 0],
    'kelaparan': [1, 0, 1, 0, 1, 0],
    'kulit_bermasalah': [0, 1, 0, 1, 0, 1],
    'infeksi_jamur': [1, 0, 1, 0, 1, 0],
    'iritasi_genital': [0, 1, 0, 1, 0, 1],
    'mudah_tersinggung': [1, 0, 1, 0, 1, 0],
    'kesemutan': [0, 1, 0, 1, 0, 1],
    'tipe_diabetes': [1, 1, 2, 0, 2, 0]
}

df = pd.DataFrame(data)

# Melatih model
X = df.drop(columns=['tipe_diabetes'])
y = df['tipe_diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Menyimpan model
joblib.dump(model, 'diabetes_model.pkl')

# Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f'Akurasi Model: {accuracy:.2f}%')
