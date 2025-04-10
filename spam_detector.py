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
    def __init__(self, model="tfidf", models_dir=None, load_if_exists=True, num_labels=2) -> None:
        """
        Initializes a Detector instance with the specified model.

        Args:
            model (str): The name of the model to use. Defaults to "tfidf".
            models_dir (Optional[str]): The directory where models are stored. If None, defaults to "models/{model_name}".
            load_if_exists (bool): Whether to load an existing model if it exists. Defaults to True.

        Raises:
            FileNotFoundError: If the specified model is not found or is invalid and loading is required.
        """
        self.model_name = model.lower()

        if models_dir is None:
            self.model_dir = os.path.join("models", self.model_name)
        else:
            self.model_dir = os.path.join(models_dir, self.model_name)

        model_exists = os.path.isdir(self.model_dir)

        if self.model_name == "tfidf":
            self.detector = TFIDFDetector()
            if model_exists and load_if_exists:
                self.detector.load(self.model_dir)

        elif model_exists and Detector.is_valid_hf_model(self.model_dir) and load_if_exists:
            self.detector = BertLikeDetector()
            self.detector.load(self.model_dir)
        
        elif not load_if_exists:
            self.detector = BertLikeDetector()
            self.detector.initialize(self.model_name, num_labels=num_labels)

        else:
            raise FileNotFoundError(f"Model '{self.model_name}' not found or is invalid")
    
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
    
    @staticmethod
    def list_models() -> list:
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
    def is_model_available(name) -> bool:
        """
        Checks if a model with the given name is available.

        Args:
            name (str): The name of the model to check.

        Returns:
            bool: True if the model is available, False otherwise.
        """

        return name == "tfidf" or Detector.is_valid_hf_model(os.path.join("models", name))

    def train(self, texts, labels, logging_dir=None, eval_texts=None, eval_labels=None, **kwargs) -> None:
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

    def evaluate(self, texts, labels) -> dict:
        """
        Evaluates the model on given texts and labels.

        Args:
            texts (list): A list of texts to evaluate.
            labels (list): A list of true labels for the given texts.

        Returns:
            The result of the evaluation from the detector, typically performance metrics.
        """
        return self.detector.evaluate(texts, labels)

    def predict(self, text_or_texts, return_proba=False) -> list:
        """
        Predicts labels for given texts.

        Args:
            text_or_texts (str or list): A text or a list of texts to predict.
            return_proba (bool): If True, returns probabilities of labels instead of labels.

        Returns:
            list: Predicted labels or probabilities of labels.
        """
        if isinstance(text_or_texts, list):
            return self.detector.predict(text_or_texts, return_proba=return_proba)
        return self.detector.predict([text_or_texts], return_proba=return_proba)[0]

    def save(self, path) -> None:
        """
        Saves the current detector model to the specified directory.

        Args:
            path (str): The directory where the model should be saved. Defaults to "models/".
        """
        self.detector.save(path)

    def load(self, path) -> None:
        """
        Loads the model from the specified directory.

        Args:
            path (str): The directory from which to load the model.
        """
        self.detector.load(path)

    def get_name(self) -> str:
        """
        Returns the name of the model.

        Returns:
            str: The name of the model.
        """
        return self.model_name
    
    def benchmark(self, texts) -> None:
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
    def __init__(self, max_features=5000, ngram_range=(1, 2)) -> None:
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

    def predict(self, texts: list, return_proba=False) -> list:
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

    def evaluate(self, texts: list, labels: list) -> dict:
        """
        Evaluates the model on given texts and labels.

        Prints the confusion matrix and classification report.

        Args:
            texts (list): A list of texts to evaluate.
            labels (list): A list of labels for given texts.

        Returns:
            dict: A dictionary containing the accuracy score, confusion matrix, and classification report.
        """
        preds = self.predict(texts)
        matrix = confusion_matrix(labels, preds)
        report = classification_report(labels, preds, output_dict=True)
        acc = accuracy_score(labels, preds)
        print("=== Confusion matrix ===")
        print(matrix)
        print("=== Report ===")
        print(report)
        print("Accuracy:", acc)
        return {"accuracy": acc, "confusion_matrix": matrix, "classification_report": report}

    def save(self, path) -> None:
        """
        Saves the model and vectorizer to the given path.

        Args:
            path (str): The directory to save the model and vectorizer to.
        """
        os.makedirs(path, exist_ok=True)

        joblib.dump(self.vectorizer, os.path.join(path, "vectorizer.pkl"))
        joblib.dump(self.model, os.path.join(path, "model.pkl"))

    def load(self, path) -> None:
        """
        Loads the model and vectorizer from the given path.

        Args:
            path (str): The directory to load the model and vectorizer from.
        """

        self.vectorizer = joblib.load(os.path.join(path, "vectorizer.pkl"))
        self.model = joblib.load(os.path.join(path, "model.pkl"))
        self.is_fitted = True

class BertLikeDetector:
    def __init__(self) -> None:
        """
        Initializes the BertLikeDetector instance.

        Sets the model, tokenizer, pipeline, and model name to None. Sets the batch size
        to 16 if a GPU is available, otherwise 4. Sets the maximum token length to 0 and
        sets the is_ready flag to False.
        """
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = None
        self.batch_size = 16 if torch.cuda.is_available() else 4
        self.max_len = 0
        self.is_ready = False
    
    def initialize(self, model_name, num_labels=2) -> None:
        """
        Initializes the BertLikeDetector with the specified model name and number of labels.

        Args:
            model_name (str): The name of the pre-trained Hugging Face model to use.
            num_labels (int): The number of labels for sequence classification. Defaults to 2.

        Sets up the tokenizer and model using the specified model name, builds the
        text classification pipeline, and marks the detector as ready.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self._build_pipeline()
        self.is_ready = True
    
    def _build_pipeline(self) -> None:
        """
        Builds a text classification pipeline for the model.

        Sets the maximum token length based on the tokenizer's model_max_length,
        adjusting to 512 if it exceeds 100,000. Initializes the pipeline with
        truncation and padding enabled and sets the device to GPU if available.

        Returns:
            None
        """
        self.max_len = self.tokenizer.model_max_length
        if self.max_len > 1e5:
            self.max_len = 512

        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            device=0 if torch.cuda.is_available() else -1,
            batch_size=self.batch_size
        )

    def train(self, texts, labels, logging_dir="/logs", epochs=3, batch_size=16 if torch.cuda.is_available() else 4, eval_texts=None, eval_labels=None) -> None:
        """
        Trains the model on given texts and labels.

        Args:
            texts (list): A list of texts to train.
            labels (list): A list of labels for given texts.
            logging_dir (str): The directory to save training logs to. Defaults to "/logs".
            epochs (int): The number of epochs to train for. Defaults to 3.
            batch_size (int): The batch size for training. Defaults to 16 if GPU is available, otherwise 4.
            eval_texts (list): A list of texts to evaluate on.
            eval_labels (list): A list of labels for given texts to evaluate on.
        """
        if not self.is_ready:
            raise ValueError("Model is not initialized or loaded.")

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

    def predict(self, texts, return_proba=False) -> list:
        """
        Predicts labels for given texts.

        Args:
            texts (list): A list of texts to predict.
            return_proba (bool): If True, returns probabilities of labels instead of labels.

        Returns:
            list: Predicted labels or probabilities of labels.
        """
        if not self.is_ready:
            raise ValueError("Model is not trained or loaded.")

        if self.pipeline is None:
            self._build_pipeline()
        
        outputs = self.pipeline(texts, batch_size=self.batch_size)
        if return_proba:
            return [
                {"label": int(out["label"].split("_")[-1]), "score": round(out["score"], 3)}
                for out in outputs
            ]
        return [int(out["label"].split("_")[-1]) for out in outputs]

    def predict_proba(self, texts) -> list:
        """
        Predicts probabilities of all labels for given texts.

        Args:
            texts (list): A list of texts to predict probabilities for.

        Returns:
            list: A list of lists, where each inner list contains probabilities of all labels for a text.
        """
        if not self.is_ready:
            raise ValueError("Model is not trained or loaded.")

        if self.pipeline is None:
            self._build_pipeline()
            
        results = self.pipeline(texts, batch_size=self.batch_size, return_all_scores=True)
        return [[score['score'] for score in output] for output in results]

    def evaluate(self, texts, labels) -> dict:
        """
        Evaluates the model on given texts and labels.

        Prints the confusion matrix and classification report.

        Args:
            texts (list): A list of texts to evaluate.
            labels (list): A list of labels for given texts.

        Returns:
            dict: A dictionary containing the accuracy score, confusion matrix, and classification report.
        """

        if not self.is_ready:
            raise ValueError("Model is not trained or loaded.")

        preds = self.predict(texts)
        matrix = confusion_matrix(labels, preds)
        report = classification_report(labels, preds, output_dict=True)
        acc = accuracy_score(labels, preds)
        print("=== Confusion matrix ===")
        print(matrix)
        print("=== Report ===")
        print(report)
        print("Accuracy:", acc)
        return {"accuracy": acc, "confusion_matrix": matrix, "classification_report": report}

    def save(self, path) -> None: 
        """
        Saves the model and tokenizer to the specified path.

        Args:
            path (str): The directory to save the model and tokenizer to.
        """
        if not self.is_ready:
            raise ValueError("Model is not trained or loaded.")

        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path) -> None:
        """
        Loads the model and tokenizer from the specified path and builds the pipeline.

        Args:
            path (str): The directory from which to load the model and tokenizer.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self._build_pipeline()
        self.is_ready = True
