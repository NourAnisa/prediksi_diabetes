from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model untuk tabel database
class UserInput(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100))
    usia = db.Column(db.Integer)
    frekuensi_buang_air_kecil = db.Column(db.Boolean)
    rasa_haus = db.Column(db.Boolean)
    kulit_mulut_kering = db.Column(db.Boolean)
    penurunan_berat_badan = db.Column(db.Boolean)
    kelelahan = db.Column(db.Boolean)
    penglihatan_buram = db.Column(db.Boolean)
    gatal_alat_kelamin = db.Column(db.Boolean)
    penyembuhan_luka_lambat = db.Column(db.Boolean)
    mata_kering = db.Column(db.Boolean)
    kelaparan = db.Column(db.Boolean)
    kulit_bermasalah = db.Column(db.Boolean)
    infeksi_jamur = db.Column(db.Boolean)
    iritasi_genital = db.Column(db.Boolean)
    mudah_tersinggung = db.Column(db.Boolean)
    kesemutan = db.Column(db.Boolean)
    prediksi = db.Column(db.String(50))

# Membuat tabel di database jika belum ada
with app.app_context():
    db.create_all()

# Memuat model yang telah dilatih
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Mengambil data dari form
    nama = request.form['nama']
    usia = request.form['usia']
    
    # Mengambil gejala dari form
    gejala = {
        'frekuensi_buang_air_kecil': int('frekuensi_buang_air_kecil' in request.form),
        'rasa_haus': int('rasa_haus' in request.form),
        'kulit_mulut_kering': int('kulit_mulut_kering' in request.form),
        'penurunan_berat_badan': int('penurunan_berat_badan' in request.form),
        'kelelahan': int('kelelahan' in request.form),
        'penglihatan_buram': int('penglihatan_buram' in request.form),
        'gatal_alat_kelamin': int('gatal_alat_kelamin' in request.form),
        'penyembuhan_luka_lambat': int('penyembuhan_luka_lambat' in request.form),
        'mata_kering': int('mata_kering' in request.form),
        'kelaparan': int('kelaparan' in request.form),
        'kulit_bermasalah': int('kulit_bermasalah' in request.form),
        'infeksi_jamur': int('infeksi_jamur' in request.form),
        'iritasi_genital': int('iritasi_genital' in request.form),
        'mudah_tersinggung': int('mudah_tersinggung' in request.form),
        'kesemutan': int('kesemutan' in request.form)
    }

    # Membuat DataFrame dari gejala
    gejala_df = pd.DataFrame([gejala])
    
    # Menambahkan kolom yang hilang dengan nilai default 0
    for column in model.feature_names_in_:
        if column not in gejala_df.columns:
            gejala_df[column] = 0
    
    # Melakukan prediksi dengan model
    prediksi = model.predict(gejala_df)[0]
    if prediksi == 1:
        hasil_prediksi = "Diabetes Tipe 1"
    elif prediksi == 2:
        hasil_prediksi = "Diabetes Tipe 2"
    else:
        hasil_prediksi = "Tidak ada diabetes"

    # Menyimpan data ke database
    new_input = UserInput(
        nama=nama,
        usia=usia,
        frekuensi_buang_air_kecil=gejala['frekuensi_buang_air_kecil'],
        rasa_haus=gejala['rasa_haus'],
        kulit_mulut_kering=gejala['kulit_mulut_kering'],
        penurunan_berat_badan=gejala['penurunan_berat_badan'],
        kelelahan=gejala['kelelahan'],
        penglihatan_buram=gejala['penglihatan_buram'],
        gatal_alat_kelamin=gejala['gatal_alat_kelamin'],
        penyembuhan_luka_lambat=gejala['penyembuhan_luka_lambat'],
        mata_kering=gejala['mata_kering'],
        kelaparan=gejala['kelaparan'],
        kulit_bermasalah=gejala['kulit_bermasalah'],
        infeksi_jamur=gejala['infeksi_jamur'],
        iritasi_genital=gejala['iritasi_genital'],
        mudah_tersinggung=gejala['mudah_tersinggung'],
        kesemutan=gejala['kesemutan'],
        prediksi=hasil_prediksi
    )
    db.session.add(new_input)
    db.session.commit()

    # Evaluasi model untuk ditampilkan
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
    X = df.drop(columns=['tipe_diabetes'])
    y = df['tipe_diabetes']
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred) * 100

    # Mengirimkan hasil prediksi dan akurasi model ke template hasil
    return render_template('result.html', nama=nama, usia=usia, prediksi=hasil_prediksi, akurasi=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
