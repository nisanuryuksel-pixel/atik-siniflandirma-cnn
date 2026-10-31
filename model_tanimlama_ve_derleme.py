# Gerekli Keras katmanlarını ve model yapısını içe aktarır.
from tensorflow.keras import layers, models

# Sequential yapısında CNN modeli oluşturur.
model = models.Sequential([
    # Giriş görüntülerinin boyutu 128x128 ve RGB formatındadır(3 renk kanalıdır).
    layers.Input(shape=(128, 128, 3)),

    # Görüntülere rastgele yatay çevirerek veri çeşitliliği sağlar.
    layers.RandomFlip("horizontal"),

    # Görüntülere hafif döndürme işlemi uygular.
    layers.RandomRotation(0.08),

    # Birinci evrişim katmanı: 32 filtre kullanarak görsel özellikleri öğrenir.
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # İkinci evrişim katmanı: 64 filtre kullanılır.
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Üçüncü evrişim katmanı: 128 filtre kullanılır.
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Özellik haritalarını tek boyutlu hâle getirir.
    layers.Flatten(),

    # 128 nöronlu tam bağlantılı katman oluşturur.
    layers.Dense(128, activation="relu"),

    # Aşırı öğrenmeyi azaltmak için yüzde %40 oranında Dropout uygulanır.
    layers.Dropout(0.4),

    # 12 farklı atık kategorisi için çıkış katmanı.
    layers.Dense(12, activation="softmax")
])

# Modelin eğitim ayarları belirlenerek derleme işlemi yapılır.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
