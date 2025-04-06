**Spamfilter** — is a simple and extensible Python library for spam detection with support for TF-IDF and transformers (BERT, DistilBERT, DeBERTa, etc.).

## Functionality

- Text classification into `spam` and `ham`
- Support for several models: TF-IDF + Logistic Regression, BERT based models, any HugginFace model. (Included pretrained models: TF-IDF + Logistic Regression, DeBERTa v3 small, DistilliBERT, RuBERT)
- Training, evaluating and benchmarking of models out of the box
- Support of pretrained models and training on user data
- Prediction, probabilities, evaluation, saving/loading
- Supports CUDA (if available) for processing

All datasets used for training the prepaired models, are listed in folder "datasets". In the sections below you will find brief descriptions of them, as well as their preparation code and comparison of models.

## Installation

```bash
git clone https://github.com/tizhproger/spamfilter.git
cd spamfilter
pip install .
```

## Datasets

<details>
  <summary>Data description</summary
  
Initial set of datasets consists of:
- **SMS** dataset. Available on: [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

Example of content:
![Dataset Content](images/sms_dataset_content.png)

![Distribution](images/sms_ds_distr.png)

<br/>

- **Twitter** dataset. Available on: [Kaggle](https://www.kaggle.com/datasets/greyhatboy/twitter-spam-dataset)

Example of content:
![Dataset Content](images/twitter_ds_content.png)

![Distribution](images/twitter_ds_distr.png)

<br/>

- **Email** dataset. Available on: [Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset)

Example of content:
![Dataset Content](images/email_ds_content.png)

![Distribution](images/email_ds_distr.png)

<br/>

- **Combined** dataset. (Joined together Email + Twitter + SMS spam datasets)

Example of content:
![Dataset Content](images/combined_ds_content.png)

![Distribution](images/combined_ds_distr.png)

<br/>

- **Combined NoEmail** dataset. (Joined together Twitter + SMS spam datasets)

Example of content:
![Dataset Content](images/noemail_ds_content.png)

![Distribution](images/noemail_ds_distr.png)

<br/>

- **Telegram** dataset. (Manually collected spam + normal messages on ru language)

Example of content:
![Dataset Content](images/telegram_ds_content.png)

![Distribution](images/telegram_ds_distr.png)

<br/>

</details>

<details>
<summary>Data preparation</summary>

But it is not that easy :)
Email dataset is a little bit harder to use, as the other ones. The reason for that is an apsence of formatting. At all.
Message text can contain Subject:, To:, Date: and other fields, which is a big noise for simple classification and even BERTa like models too.
To fix it, I used a simple cleaning code to remove all unnecessary text pieces. (It costed a text sence in some messages, but it is better than nothing)

Here is the code I used to cleanup Email dataset:
```python
import pandas as pd
import re

df_email = pd.read_csv("drive/MyDrive/ml_data/emails_dataset.csv", encoding="latin-1")

def clean_email_text(text: str) -> str:
    text = re.sub(r'^(Subject|From|To|Date)\s*:\s*', '', text) # Removing frequent texts
    text = re.sub(r'<[^>]+>', '', text) # Remove HTML tags too
    text = re.sub(r'[^a-z0-9\s\.\,\!\?\:\-]', ' ', text) # Remove any unnecessary symbol
    text = re.sub(r'\s+', ' ', text) # Merge multiple spaces into one
    text = text.strip()
    return text

df_email = df_email[['spam', 'text']]
df_email.columns = ['label', 'message']
df_email['message'] = df_email['message'].apply(clean_email_text)
df_email.head()
```

</details>

## Models Evaluation Report

## Goal

Evaluate the behavior of classification models on different datasets, in task of spam detection. Test impact of standalone and combined datasets, e.g. after excluding data to test the impact of potential noise and overfitting.

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
