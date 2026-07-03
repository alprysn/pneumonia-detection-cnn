# Derin Öğrenme Tabanlı Akciğer Röntgeninden Zatürre Tespiti

Bu proje, akciğer röntgen görüntüsünü analiz ederek görüntüyü **Normal** veya **Zatürre** olarak sınıflandıran CNN tabanlı bir karar destek prototipidir.


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


```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```


```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



```text
data/chest_xray/train/NORMAL
data/chest_xray/train/PNEUMONIA
data/chest_xray/val/NORMAL
data/chest_xray/val/PNEUMONIA
data/chest_xray/test/NORMAL
data/chest_xray/test/PNEUMONIA
```


```bash
python train_model.py --data_dir data/chest_xray --epochs 10
```


- `model.h5`
- `class_names.json`
- `grafikler/accuracy_loss.png`
- `grafikler/confusion_matrix.png`


```bash
streamlit run app.py
```

Tarayıcıda açılan ekrandan röntgen görüntüsü yüklenir. Sistem sonucu **Normal** veya **Zatürre** olarak gösterir.


