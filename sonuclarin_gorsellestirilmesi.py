# Grafik oluşturmak için gerekli kütüphaneyi kullanır.
plt.figure(figsize=(12, 5))

# Doğruluk grafiği oluşturulur.
plt.subplot(1, 2, 1)

# Eğitim ve doğrulama doğruluk değerlerini çizer.
plt.plot(history.history["accuracy"], label="Eğitim")
plt.plot(history.history["val_accuracy"], label="Doğrulama")

plt.title("Model Doğruluğu")  # Grafiğin başlığını belirler.
plt.xlabel("Epoch")  # Yatay ekseni adlandırır.
plt.ylabel("Accuracy")  # Dikey ekseni adlandırır.
plt.legend()  # Grafik çizgilerinin açıklamalarını gösterir.

# Kayıp grafiği oluşturulur.
plt.subplot(1, 2, 2)

# Eğitim ve doğrulama kayıp değerlerini çizer.
plt.plot(history.history["loss"], label="Eğitim")
plt.plot(history.history["val_loss"], label="Doğrulama")

plt.title("Model Kayıp Değerleri")  # Grafiğin başlığını belirler.
plt.xlabel("Epoch")  # Yatay ekseni adlandırır.
plt.ylabel("Loss")  # Dikey ekseni adlandırır.
plt.legend()  # Grafik çizgilerinin açıklamalarını gösterir.

# Grafiklerin düzenini ayarlar.
plt.tight_layout()

# Oluşturulan grafikleri ekranda gösterir.
plt.show()
