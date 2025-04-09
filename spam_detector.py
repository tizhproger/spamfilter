import joblib
import os
import torch
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import TextClassificationPipeline
from datasets import Dataset
from utils import free_gpu, diagnostic_report

torch.cuda.empty_cache()

class Detector:
    def __init__(self, model="tfidf", model_dir=None, load_if_exists=True):
        """
        Initializes a Detector with the specified model.

        Args:
            model (str): The model to use. Defaults to "tfidf".
            model_dir (str): The directory where the model is saved. Defaults to None.
            load_if_exists (bool): If True and the model exists, loads the model. Defaults to True.

        Raises:
            FileNotFoundError: If the model does not exist and load_if_exists is True.
        """
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
        """Checks if a directory contains a valid Hugging Face model.

        Args:
            path (str): The path to the directory to check.

        Returns:
            bool: True if the directory contains a valid Hugging Face model, False otherwise.
        """
        required_files = ["config.json", "pytorch_model.bin"]
        return all(os.path.isfile(os.path.join(path, f)) for f in required_files)
    
    @classmethod
    def custom(cls, model_name, save_as=None):
        """Creates a Detector with a custom Hugging Face model.

        Args:
            model_name (str): The name of the Hugging Face model to use.
            save_as (str, optional): The directory to save the model to. Defaults to None.

        Returns:
            Detector: A Detector instance with the specified model.
        """
        path = os.path.join("models", save_as or model_name.replace("/", "_"))
        return cls(model=model_name, model_dir=path, load_if_exists=False)
    
    @staticmethod
    def list_models():
        """Lists all available models. Returns a list of strings, where each string is
        the name of a model. The list includes "tfidf" if the TF-IDF model is available.
        """
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
        """
        Checks if a model with the given name is available.

        Args:
            name (str): The name of the model to check.

        Returns:
            bool: True if the model is available, False otherwise.
        """

        return name == "tfidf" or Detector.is_valid_hf_model(os.path.join("models", name))

    def train(self, texts, labels, logging_dir=None, eval_texts=None, eval_labels=None, **kwargs):
        """
        Trains the model on given texts and labels.

        Args:
            texts (list): A list of texts to train.
            labels (list): A list of labels for given texts.
            logging_dir (str): The directory to save training logs to. Defaults to "/logs".
            eval_texts (list): A list of texts to evaluate on.
            eval_labels (list): A list of labels for given texts to evaluate on.
            **kwargs: Additional keyword arguments passed to the Hugging Face Trainer.
        """
        free_gpu()
        diagnostic_report(texts, labels)
        self.detector.train(texts, labels, logging_dir=logging_dir or "/logs", eval_texts=eval_texts, eval_labels=eval_labels, **kwargs)

    def evaluate(self, texts, labels):
        """
        Evaluates the model on given texts and labels.

        Args:
            texts (list): A list of texts to evaluate.
            labels (list): A list of true labels for the given texts.

        Returns:
            The result of the evaluation from the detector, typically performance metrics.
        """
        return self.detector.evaluate(texts, labels)

    def predict(self, text_or_texts, return_proba=False):
        """
        Predicts the label for a single text or a list of texts.

        Args:
            text_or_texts (str or list): A single text or a list of texts to predict the labels for.

        Returns:
            int: The predicted label for the given text(s).
        """
        if isinstance(text_or_texts, list):
            return self.detector.predict(text_or_texts, return_proba=return_proba)
        return self.detector.predict([text_or_texts], return_proba=return_proba)[0]

    def predict_batch(self, texts):
        """
        Predicts the labels for a list of texts.

        Args:
            texts (list): A list of strings to predict the labels for.

        Returns:
            list: A list of predicted labels, one for each text in the input list.
        """
        return [self.predict(t) for t in texts]

    def save(self, path="models/"):
        """
        Saves the current detector model to the specified directory.

        Args:
            path (str): The directory where the model should be saved. Defaults to "models/".
        """
        model_path = os.path.join(path, self.model_name)
        self.detector.save(model_path)

    def load(self, path="models/"):
        """
        Loads the model from the specified directory.

        Args:
            path (str): The directory from which to load the model. Defaults to "models/".
        """
        model_path = os.path.join(path, self.model_name)
        self.detector.load(model_path)

    def get_name(self):
        """
        Returns the name of the model.

        Returns:
            str: The name of the model.
        """
        return self.model_name
    
    def benchmark(self, texts):
        """
        Runs a benchmark of the model on the given list of texts.

        Prints the time taken to make predictions on the given texts.

        Args:
            texts (list): The list of texts to benchmark on.
        """
        start = time.time()
        self.predict(texts)
        print(f"Time for {len(texts)} predictions: {time.time() - start:.2f} sec")

class TFIDFDetector:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initializes the TFIDFDetector with a TfidfVectorizer and LogisticRegression model.

        Args:
            max_features (int): The maximum number of features to use. Default is 5000.
            ngram_range (tuple): The range of n-values for different n-grams to be extracted. Default is (1, 2).
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)
        self.is_fitted = False

    def train(self, texts: list, labels: list, **kwargs) -> None:
        """
        Trains the model on given texts and labels.

        Args:
            texts (list): A list of texts to train.
            labels (list): A list of labels for given texts.
        """

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_fitted = True

    def predict(self, texts: list, return_proba=False):
        """
        Predicts labels for given texts.

        Args:
            texts (list): A list of texts to predict.
            return_proba (bool): If True, returns probabilities of labels instead of labels.

        Returns:
            list: Predicted labels or probabilities of labels.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained or loaded.")
        
        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)

        if return_proba:
            probs = self.model.predict_proba(X).max(axis=1)
            return [{"label": int(p), "score": round(s, 3)} for p, s in zip(preds, probs)]

        return preds.tolist()

    def predict_proba(self, texts: list):
        """
        Predicts probabilities of labels for given texts.

        Args:
            texts: list of strings

        Returns:
            numpy array of shape (n_samples, n_classes) with probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained or loaded.")
        
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def evaluate(self, texts: list, labels: list) -> None:
        """
        Evaluates the model on given texts and labels.

        Prints the confusion matrix and classification report.

        Args:
            texts (list): A list of texts to evaluate.
            labels (list): A list of labels for given texts.
        """
        y_pred = self.predict(texts)
        print("Confusion matrix:\n", confusion_matrix(labels, y_pred))
        print("\nClassification report:\n", classification_report(labels, y_pred))

    def save(self, path="models/") -> None:
        """
        Saves the model and vectorizer to the given path.

        Args:
            path (str): The directory to save the model and vectorizer to.
        """
        os.makedirs(path, exist_ok=True)

        joblib.dump(self.vectorizer, os.path.join(path, "vectorizer.pkl"))
        joblib.dump(self.model, os.path.join(path, "model.pkl"))

    def load(self, path="models/") -> None:
        """
        Loads the model and vectorizer from the given path.

        Args:
            path (str): The directory to load the model and vectorizer from.
        """

        self.vectorizer = joblib.load(os.path.join(path, "vectorizer.pkl"))
        self.model = joblib.load(os.path.join(path, "model.pkl"))
        self.is_fitted = True

class BertLikeDetector:
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initializes a BertLikeDetector with the specified model.

        Args:
            model_name (str): The name of the model to use. Defaults to "distilbert-base-uncased".
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.pipeline = None
        self.is_trained = False
        self.batch_size = 16 if torch.cuda.is_available() else 4
        self.max_len = self.tokenizer.model_max_length
        if self.max_len > 1e5:
            self.max_len = 512

    def train(self, texts, labels, logging_dir="/logs", epochs=3, batch_size=8, eval_texts=None, eval_labels=None):
        """
        Trains the model on given texts and labels.

        Args:
            texts (list): A list of texts to train.
            labels (list): A list of labels for given texts.
            logging_dir (str): The directory to save training logs to. Defaults to "/logs".
            epochs (int): The number of epochs to train. Defaults to 3.
            batch_size (int): The batch size to use for training. Defaults to 8 if GPU is available, otherwise 4.
            eval_texts (list): A list of texts to evaluate on.
            eval_labels (list): A list of labels for given texts to evaluate on.

        Returns:
            None
        """
        dataset = Dataset.from_dict({"text": texts, "label": labels})

        def tokenize(example):
            return self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len)
        
        tokenized = dataset.map(tokenize, batched=True)

        eval_dataset = None
        if eval_texts is not None and eval_labels is not None:
            eval_data = Dataset.from_dict({"text": eval_texts, "label": eval_labels})
            eval_dataset = eval_data.map(tokenize, batched=True)

        args = TrainingArguments(
            eval_strategy="epoch" if eval_dataset is not None else "no",
            per_device_train_batch_size=batch_size if batch_size else self.batch_size,
            per_device_eval_batch_size=batch_size if batch_size else self.batch_size,
            num_train_epochs=epochs,
            logging_dir=logging_dir,
            save_strategy="no",
            load_best_model_at_end=False
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )

        trainer.train()

        if torch.cuda.is_available():
            print(f"📈 After train start: allocated = {torch.cuda.memory_allocated() / 1024**2:.2f} MB, reserved = {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        self.is_trained = True

    def predict(self, texts, return_proba=False):
        """
        Predicts labels for given texts.

        Args:
            texts (list): A list of texts to predict.
            return_proba (bool): If True, returns probabilities of labels instead of labels.

        Returns:
            list: Predicted labels or probabilities of labels.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained or loaded.")

        if self.pipeline is None:
            self.pipeline = TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=True,
                padding=True,
                max_length=self.max_len,
                device=0 if torch.cuda.is_available() else -1,
                batch_size=self.batch_size)
        
        outputs = self.pipeline(texts, batch_size=self.batch_size)
        if return_proba:
            return [
                {"label": int(out["label"].split("_")[-1]), "score": round(out["score"], 3)}
                for out in outputs
            ]
        return [int(out["label"].split("_")[-1]) for out in outputs]

    def predict_proba(self, texts):
        """
        Predicts probabilities of all labels for given texts.

        Args:
            texts (list): A list of texts to predict probabilities for.

        Returns:
            list: A list of lists, where each inner list contains probabilities of all labels for a text.
        """
        if self.pipeline is None:
            self.pipeline = TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=True,
                padding=True,
                max_length=self.max_len,
                device=0 if torch.cuda.is_available() else -1,
                batch_size=self.batch_size,
                return_all_scores=True)
            
        results = self.pipeline(texts, batch_size=self.batch_size, return_all_scores=True)
        return [[score['score'] for score in output] for output in results]

    def evaluate(self, texts, labels):
        """
        Evaluates the model on given texts and labels.

        Prints the confusion matrix and classification report.

        Args:
            texts (list): A list of texts to evaluate.
            labels (list): A list of labels for given texts.

        Returns:
            list: Predicted labels.
        """
        preds = self.predict(texts)
        print("=== Confusion matrix ===")
        print(confusion_matrix(labels, preds))
        print("=== Report ===")
        print(classification_report(labels, preds))
        print("Accuracy:", accuracy_score(labels, preds))
        return preds

    def save(self, path="models/"): 
        """
        Saves the model and tokenizer to the specified path.

        Args:
            path (str): The directory to save the model and tokenizer to. Defaults to "models/".
        """
        if not os.path.exists(path):
            os.makedirs(path)

        full_path = os.path.join(path, self.model_name)
        self.model.save_pretrained(full_path)
        self.tokenizer.save_pretrained(full_path)

    def load(self, path="models/"):
        """
        Loads the model and tokenizer from the specified path and initializes the pipeline.

        Args:
            path (str): The directory from which to load the model and tokenizer. Defaults to "models/".

        Raises:
            ValueError: If the model or tokenizer cannot be loaded from the specified path.
        """
        full_path = os.path.join(path, self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(full_path)
        self.tokenizer = AutoTokenizer.from_pretrained(full_path)
        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            device=0 if torch.cuda.is_available() else -1,
            batch_size=self.batch_size)
        
        self.is_trained = True
