# Fashion-MNIST ile Kıyafet Sınıflandırma

Derin öğrenme projesi: 28×28 gri tonlamalı kıyafet görüntülerini 10 sınıfa ayıran CNN, klasik ML ve MLP taban çizgileriyle karşılaştırması, veri artırma, istatistiksel anlamlılık testi, olasılık kalibrasyonu ve Grad-CAM ile hata analizi.

- `fashion_mnist_cnn.ipynb`: çalıştırılmış not defteri (tüm çıktılar içinde)
- `Rapor.docx`: 4 sayfalık proje raporu
- `figures/`, `metrics.json`: not defterinin ürettiği grafikler ve sayısal sonuçlar
- `ek_ml_meme_kanseri/`: ek çalışma: UCI Breast Cancer Wisconsin üzerinde Logistic Regression ve Random Forest karşılaştırması

## Sonuçlar (test seti, 10.000 görüntü; derin modeller 3 tohum ortalaması)
| Model | Accuracy |
|---|---|
| Logistic Regression | 0,845 |
| Random Forest | 0,875 |
| MLP | 0,880 |
| **CNN** | **0,908** |
| CNN + veri artırma | 0,888 |

- Veri artırma bu eğitim bütçesinde (16 epoch) doğruluğu artırmadı; McNemar testine göre fark anlamlı ve artırmasız CNN lehine (p < 0,001). Artırmalı model yakınsamamış görünüyor.
- En zor sınıf Gömlek (hata oranı ≈ %31); en sık karışmalar Gömlek ↔ Tişört/Ceket/Kazak.
- Kalibrasyon iyi (ECE ≈ 0,008). Güven ≥ 0,90 olan tahminler testin ≈ %76'sını kapsıyor ve doğruluğu ≈ %98.
- Eğitim CPU'da yapıldığı için epoch sayısı kısa tutuldu (CNN 8, artırmalı CNN 16); daha uzun eğitimle sonuçlar iyileşebilir.

## Çalıştırma
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace fashion_mnist_cnn.ipynb
```
Veri: Zalando Research, Fashion-MNIST (MIT) https://github.com/zalandoresearch/fashion-mnist
