**Spamfilter** — это простая и расширяемая Python-библиотека для обнаружения спама с поддержкой TF-IDF и трансформеров (BERT, DistilBERT, DeBERTa и др.).

## 🚀 Возможности

- Классификация текста на `spam` и `ham`
- Поддержка нескольких моделей:
  - TF-IDF + Logistic Regression
  - Любая HuggingFace-модель (DeBERTa, BERT, RoBERTa, RuBERT и др.)
- Обучение на пользовательских данных
- Предсказание, вероятности, оценка, сохранение/загрузка
- Автоматическая работа с CUDA (если доступна)

## 🔧 Установка

```bash
git clone https://github.com/yourusername/spamfilter.git
cd spamfilter
pip install .
```


# Spam Detection Models Evaluation Report

## Goal

Evaluate the behavior of spam detection models after excluding email messages to test the impact of potential noise.

---

## Datasets

- **Combined**: Email + SMS + Twitter
- **Filtered**: SMS + Twitter only

---

## Results

### TF-IDF + Logistic Regression

| Dataset   | Accuracy | Recall (Spam) | Precision (Spam) | F1 (Spam) | FP | FN |
|-----------|----------|----------------|------------------|-----------|----|----|
| Full      | 0.9733   | 0.93           | 0.91             | 0.92      | 52 | 38 |
| No Emails | 0.9874   | 0.97           | 0.94             | 0.95      | 19 | 9  |

### DeBERTa

| Dataset   | Accuracy | Recall (Spam) | Precision (Spam) | F1 (Spam) | FP | FN |
|-----------|----------|----------------|------------------|-----------|----|----|
| Full      | 0.9914   | 0.95           | 1.00             | 0.97      | 1  | 28 |
| No Emails | 0.9996   | 1.00           | 1.00             | 1.00      | 1  | 0  |

---

## Visualizations

### Metric Comparison

![Metric Comparison](metrics_comparison.png)

### Confusion Errors (FP + FN)

![Confusion Matrix Comparison](confusion_comparison.png)

---

## Findings

- **Email data introduces noise**, increasing false negatives and lowering recall.
- **TF-IDF** is more sensitive to structured noise (e.g., headers, HTML) in email content, but still effective.
- **DeBERTa** consistently delivers high performance across datasets, achieving near-perfect results when trained without noisy email data.

---

## Recommendation

- For simplicity and reproducibility, **TF-IDF + LogisticRegression** remains a solid option.
- For **highest accuracy and robustness**, especially across diverse text formats, **DeBERTa is the better choice**.
