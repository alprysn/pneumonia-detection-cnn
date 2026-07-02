import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ], name="data_augmentation")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.Rescaling(1.0 / 255.0),
        data_augmentation,
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_dataset(base_dir: Path):
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    test_dir = base_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Train klasörü bulunamadı: {train_dir}\n"
            "Beklenen yapı: data/chest_xray/train/NORMAL ve data/chest_xray/train/PNEUMONIA"
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
    )

    val_ds = None
    if val_dir.exists():
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
            shuffle=False,
        )

    test_ds = None
    if test_dir.exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
            shuffle=False,
        )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    if val_ds:
        val_ds = val_ds.prefetch(buffer_size=autotune)
    if test_ds:
        test_ds = test_ds.prefetch(buffer_size=autotune)
    return train_ds, val_ds, test_ds, class_names


def plot_history(history, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("accuracy", []), label="Eğitim accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Doğrulama accuracy")
    plt.plot(history.history.get("loss", []), label="Eğitim loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Doğrulama loss")
    plt.title("Model Eğitim Performansı")
    plt.xlabel("Epoch")
    plt.ylabel("Değer")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_loss.png", dpi=200)
    plt.close()


def evaluate_model(model, test_ds, out_dir: Path):
    if test_ds is None:
        print("Test klasörü bulunamadı; confusion matrix üretilmedi.")
        return

    y_true = []
    y_pred = []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).reshape(-1)
        preds = (probs >= 0.5).astype(int)
        y_true.extend(labels.numpy().reshape(-1).astype(int).tolist())
        y_pred.extend(preds.tolist())

    print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))
    cm = confusion_matrix(y_true, y_pred)

    out_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.xticks([0, 1], ["Normal", "Zatürre"])
    plt.yticks([0, 1], ["Normal", "Zatürre"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/chest_xray", help="İçinde train/val/test klasörleri olan veri seti yolu")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    train_ds, val_ds, test_ds, class_names = load_dataset(base_dir)

    with open("class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("model.h5", monitor="val_accuracy" if val_ds else "accuracy", save_best_only=True, mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss" if val_ds else "loss", patience=3, restore_best_weights=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save("model.h5")
    plot_history(history, Path("grafikler"))
    evaluate_model(model, test_ds, Path("grafikler"))
    print("Eğitim tamamlandı. model.h5, grafikler/accuracy_loss.png ve varsa confusion_matrix.png oluşturuldu.")


if __name__ == "__main__":
    main()
