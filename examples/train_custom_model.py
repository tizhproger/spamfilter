from spamfilter import Detector
import pandas as pd

df = pd.read_csv("your_dataset.csv")
texts, labels = df["text"].tolist(), df["label"].tolist()
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

model_name = "microsoft/deberta-v3-small"
save_as = "models/rubert_custom"

detector = Detector(model_name, load_if_exists=False)
detector.train(X_train, y_train, eval_texts = X_test, eval_labels = y_test)
detector.save(save_as)
