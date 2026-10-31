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

# Karmaşıklık matrisi oluşturmak için gerekli kütüphaneleri içe aktarır.
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Gerçek sınıflar ve modelin tahminleri kullanılarak karmaşıklık matrisi oluşturulur.
cm = confusion_matrix(y_true, y_pred)

# Karmaşıklık matrisinin grafik boyutunu belirler.
plt.figure(figsize=(12, 10))

# Karmaşıklık matrisini ısı haritası şeklinde görselleştirir.
sns.heatmap(
    cm,
    annot=True,  # Hücrelerdeki değerleri gösterir.
    fmt="d",  # Değerleri tam sayı olarak gösterir.
    cmap="Blues",  # Mavi renk tonlarını kullanır.
    xticklabels=class_names,  # Yatay eksende tahmin edilen sınıfları gösterir.
    yticklabels=class_names  # Dikey eksende gerçek sınıfları gösterir.
)

plt.xlabel("Tahmin Edilen Sınıf")  # Yatay eksenin adını belirler.
plt.ylabel("Gerçek Sınıf")  # Dikey eksenin adını belirler.
plt.title("Atık Sınıflandırma Karmaşıklık Matrisi")  # Grafik başlığını belirler.
plt.xticks(rotation=45)  # Yatay eksen etiketlerini 45 derece döndürür.
plt.yticks(rotation=0)  # Dikey eksen etiketlerini yatay tutar.
plt.tight_layout()  # Grafik düzenini ayarlar.
plt.show()  # Grafiği ekranda gösterir.
