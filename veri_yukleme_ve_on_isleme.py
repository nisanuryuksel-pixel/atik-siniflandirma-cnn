# Google Colab üzerinden bilgisayardan dosya yüklemek için kullanılır.
from google.colab import files

# Veri setinin Colab ortamına yüklenmesini sağlar.
uploaded = files.upload()

# ZIP dosyalarını açmak için gerekli kütüphaneler.
import zipfile
import os

# Yüklenecek ZIP dosyasının konumu.
zip_path = "/content/archive.zip"

# ZIP dosyasının çıkarılacağı klasör.
extract_path = "/content/garbage_dataset"

# ZIP dosyasını açarak içindeki dosyaları belirtilen klasöre çıkarır.
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# İşlemin tamamlandığını bildirir.
print("Veri seti başarıyla çıkarıldı!")
