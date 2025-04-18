import pandas as pd
import time
from spam_detector import Detector
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Be aware, thatyour dataset for benchmark should have two columns: 'message' and 'label', if you want to use this code as it is.
# You can change it according to your need on line 62; namely:
# Here: texts, labels = df["message"].tolist(), df["label"].tolist()

def evaluate_model(full_model_name, path_name, texts, labels, dataset_name, save=True):
    model_path = f"models/{path_name}"

    model_names = {
        "tfidf": "TF-IDF + LR",
        "distilbert": "DistilBERT",
        "deberta": "DeBERTa v3 Small",
        "bert": "BERT Base",
        "roberta": "RoBERTa Base",
        #"rubert": "RuBERT Tiny", # use ONLY with RU dataset
        "xlm_roberta": "XLM-RoBERTa Base",
        "bert_multi": "BERT Multilingual",
        "distilbert_multi": "DistilBERT Multilingual",
    }

    print(f"🔧 Training new model: {model_names[path_name]}")
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    detector = Detector(full_model_name, load_if_exists=False)
    detector.train(X_train, y_train, eval_texts=X_test, eval_labels=y_test)
    if save:
        detector.save(model_path)

    # Prediction time
    start = time.time()
    preds = detector.predict(X_test)
    duration = time.time() - start

    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    cm = confusion_matrix(y_test, preds)
    fp = cm[0][1]
    fn = cm[1][0]

    return {
        "Model": model_names[path_name],
        "Dataset": dataset_name,
        "Accuracy": round(acc, 4),
        "Recall (Spam)": round(rec, 2),
        "Precision (Spam)": round(prec, 2),
        "F1 (Spam)": round(f1, 2),
        "FP": int(fp),
        "FN": int(fn),
        "Time (s)": round(duration, 2)
    }

def benchmark_all(datasets, models):
    results = []

    for dataset_name, dataset_path in datasets.items():
        print(f"\n📊 Dataset: {dataset_name}")
        df = pd.read_csv(dataset_path)
        texts, labels = df["message"].tolist(), df["label"].tolist()

        for path_name, full_model_name in models.items():
            result = evaluate_model(full_model_name, path_name, texts, labels, dataset_name, save=False)
            results.append(result)

    return pd.DataFrame(results)

if __name__ == "__main__":
    datasets = {
        "Combined (Email + Twitter + SMS)": "./spam_datasets/combined_dataset.csv",
        "Combined NoEmail (Twitter + SMS)": "./spam_datasets/combined_noemail_dataset.csv",
        "SMS": "./spam_datasets/processed_sms.csv",
        "Twitter": "./spam_datasets/processed_twitter.csv",
        "Email": "./spam_datasets/processed_email.csv",
        #"Telegram": "./spam_datasets/telegram_spam_ru.csv" # use ONLY with RU/Multilingual models
    }

    models = {
        "tfidf": "tfidf",
        "distilbert": "distilbert-base-uncased",
        "deberta": "microsoft/deberta-v3-small",
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        #"rubert": "cointegrated/rubert-tiny2", # use ONLY with RU dataset
        "xlm_roberta": "xlm-roberta-base",
        "bert_multi": "bert-base-multilingual-cased",
        "distilbert_multi": "distilbert-base-multilingual-cased"}

    df_results = benchmark_all(datasets, models)
    df_results.to_csv("benchmark_results.csv", index=False)
    print("\n✅ Benchmark complete. Results saved to benchmark_results.csv")
