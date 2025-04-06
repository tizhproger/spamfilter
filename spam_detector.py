import joblib
import os
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import TextClassificationPipeline
from datasets import Dataset

class Detector:
    def __init__(self, model="tfidf", model_dir=None, load_if_exists=True):
        self.model_name = model.lower()
        self.model_dir = model_dir or os.path.join("models", self.model_name)

        model_exists = os.path.isdir(self.model_dir)

        if self.model_name == "tfidf":
            self.detector = TFIDFDetector()
            if model_exists and load_if_exists:
                self.detector.load(self.model_dir)

        elif model_exists and Detector.is_valid_hf_model(self.model_dir) and load_if_exists:
            self.detector = BertLikeDetector(model_name=model_dir)
            self.detector.load(self.model_dir)
        
        elif not load_if_exists:
            self.detector = BertLikeDetector(model_name=self.model_name)

        else:
            raise FileNotFoundError(f"Model directory '{self.model_dir}' not found or is invalid")
    
    @staticmethod
    def is_valid_hf_model(path: str) -> bool:
        required_files = ["config.json", "pytorch_model.bin"]
        return all(os.path.isfile(os.path.join(path, f)) for f in required_files)
    
    @classmethod
    def custom(cls, model_name, save_as=None):
        """
        Создаёт BertLikeDetector с кастомным Hugging Face моделью.
        save_as — имя папки для сохранения (например, 'my_rubert')
        """
        path = os.path.join("models", save_as or model_name.replace("/", "_"))
        return cls(model=model_name, model_dir=path, load_if_exists=False)
    
    @staticmethod
    def list_models():
        base_dir = "models"
        if not os.path.isdir(base_dir):
            return []
        return [
            name for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name)) and
            Detector.is_valid_hf_model(os.path.join(base_dir, name)) or name == "tfidf"
        ]
    
    @staticmethod
    def is_model_available(name):
        return name == "tfidf" or Detector.is_valid_hf_model(os.path.join("models", name))

    def train(self, texts, labels, output_dir="model_output", **kwargs):
        self.detector.train(texts, labels, output_dir=output_dir, **kwargs)

    def evaluate(self, texts, labels):
        return self.detector.evaluate(texts, labels)

    def predict(self, text_or_texts):
        if isinstance(text_or_texts, list):
            return self.detector.predict(text_or_texts)
        return self.detector.predict([text_or_texts])[0]

    def predict_batch(self, texts):
        return [self.predict(t) for t in texts]

    def save(self, path="models/"):
        model_path = os.path.join(path, self.model_name)
        self.detector.save(model_path)

    def load(self, path="models/"):
        model_path = os.path.join(path, self.model_name)
        self.detector.load(model_path)

    def get_name(self):
        return self.model_name
    
    def benchmark(self, texts):
        import time
        start = time.time()
        self.predict(texts)
        print(f"Time for {len(texts)} predictions: {time.time() - start:.2f} sec")

class TFIDFDetector:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)
        self.is_fitted = False

    def train(self, texts: list, labels: list, **kwargs) -> None:
        """
        Обучает векторизатор и модель на новых данных.
        """
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_fitted = True

    def predict(self, texts: list):
        """
        Предсказывает метки для новых сообщений.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained or loaded.")
        
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts: list):
        """
        Возвращает вероятности классов.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained or loaded.")
        
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def evaluate(self, texts: list, labels: list) -> None:
        """
        Оценивает модель: accuracy, precision, recall, f1, confusion matrix.
        """
        y_pred = self.predict(texts)
        print("Confusion matrix:\n", confusion_matrix(labels, y_pred))
        print("\nClassification report:\n", classification_report(labels, y_pred))

    def save(self, path="models/") -> None:
        """
        Сохраняет модель и векторизатор в папку.
        """
        os.makedirs(path, exist_ok=True)

        joblib.dump(self.vectorizer, os.path.join(path, "vectorizer.pkl"))
        joblib.dump(self.model, os.path.join(path, "model.pkl"))

    def load(self, path="models/") -> None:
        """
        Загружает модель и векторизатор из папки.
        """
        self.vectorizer = joblib.load(os.path.join(path, "vectorizer.pkl"))
        self.model = joblib.load(os.path.join(path, "model.pkl"))
        self.is_fitted = True

class BertLikeDetector:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.pipeline = None
        self.is_trained = False

    def train(self, texts, labels, output_dir="bert_model", epochs=3, batch_size=8):
        dataset = Dataset.from_dict({"text": texts, "label": labels})
        dataset = dataset.train_test_split(test_size=0.2)

        def tokenize(example):
            return self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length)
        
        tokenized = dataset.map(tokenize, batched=True)

        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_dir=f"{output_dir}/logs",
            save_strategy="no",
            load_best_model_at_end=False
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            tokenizer=self.tokenizer
        )

        trainer.train()
        self.is_trained = True

    def predict(self, texts):
        if not self.is_trained:
            raise ValueError("Model is not trained or loaded.")

        if self.pipeline is None:
            self.pipeline = TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=True,
                padding=True,
                max_length=self.tokenizer.model_max_length,
                device=0 if torch.cuda.is_available() else -1)
            
        outputs = self.pipeline(texts)
        return [int(o["label"].split("_")[-1]) for o in outputs]

    def predict_proba(self, texts):
        if self.pipeline is None:
            self.pipeline = TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=True,
                padding=True,
                max_length=self.tokenizer.model_max_length,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True)
            
        results = self.pipeline(texts, return_all_scores=True)
        return [[score['score'] for score in output] for output in results]

    def evaluate(self, texts, labels):
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        preds = self.predict(texts)
        print("=== Confusion matrix ===")
        print(confusion_matrix(labels, preds))
        print("=== Report ===")
        print(classification_report(labels, preds))
        print("Accuracy:", accuracy_score(labels, preds))
        return preds

    def save(self, path="models/"): 
        if not os.path.exists(path):
            os.makedirs(path)

        full_path = os.path.join(path, self.model_name)
        self.model.save_pretrained(full_path)
        self.tokenizer.save_pretrained(full_path)

    def load(self, path="models/"):
        full_path = os.path.join(path, self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(full_path)
        self.tokenizer = AutoTokenizer.from_pretrained(full_path)
        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            padding=True,
            max_length=self.tokenizer.model_max_length,
            device=0 if torch.cuda.is_available() else -1)
        
        self.is_trained = True
