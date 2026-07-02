# Derin Öğrenme Tabanlı Akciğer Röntgeninden Zatürre Tespiti

Bu proje, akciğer röntgen görüntüsünü analiz ederek görüntüyü **Normal** veya **Zatürre** olarak sınıflandıran CNN tabanlı bir karar destek prototipidir.

> Uyarı: Bu sistem klinik tanı koymaz. Eğitim amaçlı karar destek prototipidir.

## Klasör Yapısı

```text
bitirme_proje_dosyasi/
├── app.py
├── train_model.py
├── requirements.txt
├── class_names_ornek.json
├── grafikler/
├── test_resimleri/
└── data/chest_xray/
    ├── train/NORMAL
    ├── train/PNEUMONIA
    ├── val/NORMAL
    ├── val/PNEUMONIA
    ├── test/NORMAL
    └── test/PNEUMONIA
```

## 1) Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Mac/Linux kullanılırsa:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Veri Seti

Kaggle'daki **Chest X-Ray Images (Pneumonia)** veri seti kullanılabilir. Veri setini indirdikten sonra klasörleri şu yapıya göre yerleştir:

```text
data/chest_xray/train/NORMAL
data/chest_xray/train/PNEUMONIA
data/chest_xray/val/NORMAL
data/chest_xray/val/PNEUMONIA
data/chest_xray/test/NORMAL
data/chest_xray/test/PNEUMONIA
```

## 3) Model Eğitimi

```bash
python train_model.py --data_dir data/chest_xray --epochs 10
```

Eğitim sonunda şu dosyalar oluşur:

- `model.h5`
- `class_names.json`
- `grafikler/accuracy_loss.png`
- `grafikler/confusion_matrix.png`

## 4) Arayüzü Çalıştırma

```bash
streamlit run app.py
```

Tarayıcıda açılan ekrandan röntgen görüntüsü yüklenir. Sistem sonucu **Normal** veya **Zatürre** olarak gösterir.

## 5) GitHub Notu

GitHub'a kodları, raporu ve küçük görselleri yükle. Büyük veri setini ve büyük `model.h5` dosyasını GitHub'a yükleme. Model dosyası gerekiyorsa Google Drive üzerinden paylaşılabilir.
