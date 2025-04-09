from spamfilter import Detector
import pandas as pd

df = pd.read_csv("your_dataset.csv")  # must contain columns: text (message), label (spam or not)
texts, labels = df["text"].tolist(), df["label"].tolist()

model_name = "cointegrated/rubert-tiny2"
save_as = "rubert_custom"

detector = Detector.custom(model_name, "models")
detector.train(texts, labels)
detector.save(save_as)
