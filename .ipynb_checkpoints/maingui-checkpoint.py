import tkinter as tk
from tkinter import filedialog, messagebox
from EmailSpam import spam_detector
from DeepImage import model_loader_img
import os
from email import policy
from email.parser import BytesParser
from PIL import Image
import tempfile

# Spam dedektörünü başlat
spam_model = spam_detector.SpamDetector()

def analyze_email():
    file_path = filedialog.askopenfilename(title="E-posta dosyası seç", filetypes=[("EML Dosyaları", "*.eml")])
    if not file_path:
        return

    try:
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)

        # Metin içeriğini al
        email_text = msg.get_body(preferencelist=('plain')).get_content()
        spam_result = spam_model.predict(email_text)
        spam_label = "SPAM" if spam_result["prediction"] == 1 else "NOT SPAM"
        spam_score = spam_result["probability"]

        # Görsel var mı? (örneğin inline img veya attachment)
        image_results = []
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type.startswith("image/"):
                img_data = part.get_payload(decode=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
                    temp_img.write(img_data)
                    temp_img_path = temp_img.name

                result = model_loader_img.predict_image(temp_img_path)
                image_results.append(f"Görsel: {result['Prediction']} (Skor: {result['Score']:.2f})")
                os.remove(temp_img_path)

        # Sonuçları göster
        final_msg = f"📧 Metin Analizi: {spam_label} (İhtimal: {spam_score:.2f})\n"
        if image_results:
            final_msg += "\n🖼️ Görsel Analizi:\n" + "\n".join(image_results)
        else:
            final_msg += "\n🖼️ Görsel bulunamadı."

        messagebox.showinfo("E-posta Analizi", final_msg)

    except Exception as e:
        messagebox.showerror("Hata", f"E-posta analizinde hata oluştu:\n{e}")

# GUI tasarımı
window = tk.Tk()
window.title("E-posta İçerik Analiz Aracı")
window.geometry("400x300")

label = tk.Label(window, text="Bir e-posta (.eml) dosyası seçin ve analiz edin.", wraplength=350)
label.pack(pady=20)

analyze_btn = tk.Button(window, text="E-posta Analiz Et", command=analyze_email, width=30)
analyze_btn.pack(pady=20)

window.mainloop()
