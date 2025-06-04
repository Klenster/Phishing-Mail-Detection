"""import pathlib
import logging
from fastai.vision.all import load_learner, PILImage
from pathlib import Path

# POSIX uyumluluğu (Colab'de Windows model dosyasını açmak için gerekebilir)
try:
    pathlib.PosixPath = pathlib.WindowsPath
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("Deepfake/model3.pkl")
MODEL = None

def load_model():
    try:
        model = load_learner(MODEL_PATH)
        model.eval()
        logger.info(f"✅ Deepfake modeli yüklendi: {MODEL_PATH.name}")
        return model
    except Exception as e:
        logger.error(f"❌ Model yüklenemedi: {str(e)}")
        raise

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL

def predict_image(img_path: str):
    try:
        model = get_model()
        img = PILImage.create(img_path)
        pred, idx, probs = model.predict(img)
        return {
            "Prediction": str(pred),
            "Possibilities": {
                str(model.dls.vocab[i]): float(probs[i]) for i in range(len(probs))
            }
        }
    except Exception as e:
        logger.error(f"❌ Tahmin hatası: {str(e)}")
        return {
            "Prediction": "error",
            "Possibilities": {"error": 1.0},
            "error_message": str(e)
     # deepfake_model.py

import tensorflow as tf
import logging
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("Deepfake/deepfake_model_deepfake.keras")
MODEL = None

def load_model():
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"✅ Deepfake modeli yüklendi: {MODEL_PATH.name}")
        return model
    except Exception as e:
        logger.error(f"❌ Model yüklenemedi: {str(e)}")
        raise

def get_model():

    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL

def predict_image(img_path: str):
  
    try:
        model = get_model()
        img = image.load_img(img_path, target_size=(224, 224))  # Resmin boyutunu modelin beklediği boyuta getir
        img_array = image.img_to_array(img)  # Resmi numpy dizisine dönüştür
        img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekleyin

        # Tahmin yap
        preds = model.predict(img_array)
        predicted_class = np.argmax(preds, axis=1)  # En yüksek olasılığı seç

        # Olasılıkları çıkar
        probs = preds[0]
        return {
            "Prediction": str(predicted_class),
            "Possibilities": {
                f"Class {i}": float(probs[i]) for i in range(len(probs))
            }
        }
    except Exception as e:
        logger.error(f"❌ Tahmin hatası: {str(e)}")
        return {
            "Prediction": "error",
            "Possibilities": {"error": 1.0},
            "error_message": str(e)
        }
   }

# deepfake_model.py
"""
#*******************************************************************************
"""
import tensorflow as tf
import logging
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path
import os
from PIL import Image  # PIL kütüphanesini unutmayalım

# Proje dizinini al (bu kodu çalıştırdığın dosyanın bulunduğu yerden)
project_dir = os.path.dirname(os.path.abspath(__file__))

# Model dosyasının göreceli yolunu oluştur
model_path = os.path.join(project_dir, 'deepfake_model_deepfake.keras')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(model_path)  # Dinamik yol ile MODEL_PATH'i ayarlıyoruz
MODEL = None

def load_keras_model():
    #Keras modelini yükleyen fonksiyon.
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"✅ Deepfake modeli yüklendi: {MODEL_PATH.name}")
        return model
    except Exception as e:
        logger.error(f"❌ Model yüklenemedi: {str(e)}")
        raise

def get_model():
    #Modeli döndüren fonksiyon.
    global MODEL
    if MODEL is None:
        MODEL = load_keras_model()  # Modeli bir defa yükle
    return MODEL

def predict_image(image_path):
    #Verilen görseli modelle tahmin et
    try:
        model = get_model()  # Modeli al

        # Görseli aç ve RGB'ye çevir
        image = Image.open(image_path).convert('RGB')

        # Yeniden boyutlandır (128x128)
        image = image.resize((128, 128))

        # NumPy dizisine çevir
        image = np.array(image)

        # Normalize et (0-255 arası -> 0-1 arası)
        image = image / 255.0

        # Batch dimension ekle: (128,128,3) -> (1,128,128,3)
        image = np.expand_dims(image, axis=0)

        # Model ile tahmin et
        prediction = model.predict(image)

        # Sonuçları kontrol et
        if prediction.shape[-1] == 1:
            score = prediction[0][0]
            label = "Deepfake" if score > 0.5 else "Real"
            possibilities={"Deepfake":score, "Real":1-score}
            logger.info(f"🧠 Tahmin: {label} ({score:.2f})")
            return {"Prediction": label, "Score": score,"Possibilities":possibilities}
        else:
            logger.warning("⚠️ Beklenmeyen tahmin şekli.")
            return {"Prediction": "Unknown", "Score": 0.0}

    except Exception as e:
        logger.error(f"❌ Tahmin hatası: {e}")
        return {"Prediction": "error", "Error": str(e)}
"""
# DeepImage/model_loader_img.py
import tensorflow as tf
import logging
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path
import os
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        try:
            base_dir = os.path.dirname(__file__)
            model_path = os.path.join(base_dir, 'deepfake_model.keras')

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"✅ Deepfake modeli yüklendi: {model_path}")
        except Exception as e:
            logger.error(f"❌ Model yüklenemedi: {e}")
            self.model = None

    def predict(self, image_path):
        if self.model is None:
            return {"Prediction": "error", "Error": "Model yüklenemedi"}

        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

            img = Image.open(image_path).convert('RGB')
            img = img.resize((150, 150))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = self.model.predict(img)

            if prediction.shape[-1] == 1:
                score = float(prediction[0][0])  # Skor sayıya dönüştürülür
                logger.info(f"Model skoru: {score:.4f}")
                label = "Deepfake" if score < 0.5 else "Real"
                possibilities = {"Deepfake": score, "Real": 1 - score}
                return {"Prediction": label, "Score": score, "Possibilities": possibilities}
            else:
                return {"Prediction": "Unknown", "Score": 0.0, "Error": "Beklenmeyen çıktı boyutu"}

        except Exception as e:
            logger.error(f"❌ Tahmin hatası: {e}")
            return {"Prediction": "error", "Error": str(e)}

