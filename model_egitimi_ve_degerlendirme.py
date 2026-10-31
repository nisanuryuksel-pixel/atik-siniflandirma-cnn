# EarlyStopping ve ModelCheckpoint yöntemlerini içe aktarır.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Doğrulama kaybını takip ederek gereksiz eğitim tekrarlarını önlemeye yardımcı olur.
early_stop = EarlyStopping(
    monitor="val_loss",  # Doğrulama kaybını takip eder.
    patience=3,  # İyileşme olmazsa 3 epoch daha bekler.
    restore_best_weights=True  # En iyi ağırlıkları geri yükler.
)

# Doğrulama kaybı en iyi olduğunda modeli kaydeder.
checkpoint = ModelCheckpoint(
    "en_iyi_model.keras",  # Kaydedilecek modelin dosya adı.
    monitor="val_loss",  # Doğrulama kaybını takip eder.
    save_best_only=True  # Yalnızca en iyi modeli kaydeder.
)

# Modelin eğitim işlemi gerçekleştirilir.
history = model.fit(
    train_data,  # Eğitim veri seti.
    validation_data=validation_data,  # Doğrulama veri seti.
    epochs=15,  # Maksimum 15 eğitim turu.
    class_weight=class_weights,  # Sınıflar arasındaki görüntü sayısı farklılıklarını dikkate alır.
    callbacks=[early_stop, checkpoint]  # Eğitim kontrollerini uygular.
)

# Eğitilen model test veri seti üzerinde değerlendirilir.
test_loss, test_accuracy = model.evaluate(test_data)

# Test doğruluğu ve kayıp değerlerini ekrana yazdırır.
print("Test doğruluğu:", test_accuracy)
print("Test kaybı:", test_loss)
