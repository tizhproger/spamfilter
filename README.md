**Spamfilter** — is a simple and extensible Python library for spam detection with support for TF-IDF and transformers (BERT, DistilBERT, DeBERTa, etc.).

## Functionality

- Text classification into `spam` and `ham`
- Support for several models: TF-IDF + Logistic Regression, BERT based models, any HugginFace model. (Included pretrained models: TF-IDF + Logistic Regression, DeBERTa v3 small, DistilBERT, RuBERT)
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
- `Processed dataset used for training is: processed_sms.csv`

Example of content:
![Dataset Content](images/sms_dataset_content.png)

![Distribution](images/sms_ds_distr.png)

<br/>

- **Twitter** dataset. Available on: [Kaggle](https://www.kaggle.com/datasets/greyhatboy/twitter-spam-dataset)
- `Processed dataset used for training is: processed_twitter.csv`

Example of content:
![Dataset Content](images/twitter_ds_content.png)

![Distribution](images/twitter_ds_distr.png)

<br/>

- **Email** dataset. Available on: [Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset)
- `Processed dataset used for training is: processed_email.csv`

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

Before using datasets we need to unify their structure, for this I used the following code:

`SMS`
```python
import pandas as pd

df_sms = pd.read_csv("drive/MyDrive/ml_data/sms_dataset.csv", encoding="latin-1")
df_sms = df_sms[['v1', 'v2']]
df_sms.columns = ['label', 'message']
df_sms['label'] = df_sms['label'].map({'ham': 0, 'spam': 1})
df_sms.head()
```

`TWITTER`
```python
import pandas as pd

df_twitter = pd.read_csv("drive/MyDrive/ml_data/twitter_dataset.csv", encoding="latin-1")
df_twitter = df_twitter[['class', 'tweets']]
df_twitter.columns = ['label', 'message']
df_twitter.head()
```

`EMAIL`

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

In order to compare how each model performs on different data, I created and ran a benchmark, which different datasets and different popular models.\
Below I provide a summaary tables for all tests. If you want to run tests by yourself, you can use `benchmark.py` script in folder `tests`.

Datasets used for testing:
- Combined `(sombined_dataset.csv)`
- Combined NoEmail `(combined_noemail_dataset.csv)`
- Twitter `(processed_twitter.csv)`
- SMS (processed_sms.csv)
- Email `(processed_email.csv)`.

`Importants notice: All datasets contain ONLY english messages`\
`TF-IDF + Logistic Regression was set with parameter class_weight='balanced', in order to compensate inbalance in datasets`\
**Time** - shows how long it took a model to predict a batch of messages.

DUE TO ERROR RESULTS BELOW ARE CURRENTLY INVALID

### BERT Base
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9994 |            1    |               1    |        1    |    1 |    9 |        146 |
| Combined NoEmail (Twitter + SMS) |     0.9996 |            1    |               1    |        1    |    0 |    4 |         78 |
| SMS                              |     0.9982 |            0.99 |               0.99 |        0.99 |    5 |    5 |         39 |
| Twitter                          |     0.9991 |            0.99 |               1    |        1    |    1 |    4 |         39 |
| Email                            |     0.9998 |            1    |               1    |        1    |    0 |    1 |         72 |

---

### BERT Multilingual
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9916 |            0.97 |               0.98 |        0.98 |   63 |   79 |        148 |
| Combined NoEmail (Twitter + SMS) |     0.9969 |            0.98 |               1    |        0.99 |    7 |   28 |         78 |
| SMS                              |     0.9946 |            0.96 |               1    |        0.98 |    3 |   27 |         39 |
| Twitter                          |     0.9975 |            0.99 |               0.99 |        0.99 |    4 |   10 |         39 |
| Email                            |     0.9977 |            0.99 |               1    |        1    |    6 |    7 |         73 |

---

### DeBERTa v3 Small
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9991 |            0.99 |               1    |        1    |    0 |   15 |        169 |
| Combined NoEmail (Twitter + SMS) |     0.9996 |            1    |               1    |        1    |    2 |    2 |         97 |
| SMS                              |     0.9978 |            0.99 |               0.99 |        0.99 |    5 |    7 |         49 |
| Twitter                          |     0.9991 |            0.99 |               1    |        1    |    1 |    4 |         50 |
| Email                            |     0.9983 |            0.99 |               1    |        1    |    2 |    8 |         72 |

---

### DistilBERT
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9992 |            1    |                  1 |        1    |    4 |   10 |         81 |
| Combined NoEmail (Twitter + SMS) |     0.9995 |            1    |                  1 |        1    |    2 |    4 |         43 |
| SMS                              |     0.9986 |            0.99 |                  1 |        0.99 |    2 |    6 |         21 |
| Twitter                          |     0.9989 |            0.99 |                  1 |        1    |    2 |    4 |         21 |
| Email                            |     0.9988 |            1    |                  1 |        1    |    1 |    6 |         39 |

---

### RoBERTa Base
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9951 |            0.97 |               1    |        0.99 |    8 |   74 |        150 |
| Combined NoEmail (Twitter + SMS) |     0.9966 |            0.98 |               1    |        0.99 |    2 |   36 |         80 |
| SMS                              |     0.9966 |            0.98 |               0.99 |        0.99 |    5 |   14 |         40 |
| Twitter                          |     0.9905 |            0.96 |               0.97 |        0.96 |   20 |   33 |         40 |
| Email                            |     0.9993 |            1    |               1    |        1    |    0 |    4 |         73 |

---

### TF-IDF + LR
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9865 |            0.98 |               0.94 |        0.96 |  179 |   48 |         10 |
| Combined NoEmail (Twitter + SMS) |     0.995  |            0.99 |               0.97 |        0.98 |   46 |   10 |          6 |
| SMS                              |     0.993  |            0.98 |               0.97 |        0.97 |   26 |   13 |          3 |
| Twitter                          |     0.9928 |            0.98 |               0.96 |        0.97 |   28 |   12 |          3 |
| Email                            |     0.9921 |            1    |               0.97 |        0.98 |   42 |    3 |          4 |

---

### XLM-RoBERTa Base
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9695 |            0.9  |               0.92 |        0.91 |  231 |  283 |        155 |
| Combined NoEmail (Twitter + SMS) |     0.995  |            0.97 |               0.99 |        0.98 |   10 |   46 |         80 |
| SMS                              |     0.9973 |            0.99 |               0.99 |        0.99 |    5 |   10 |         40 |
| Twitter                          |     0.9964 |            0.98 |               1    |        0.99 |    3 |   17 |         41 |
| Email                            |     0.9883 |            0.98 |               0.98 |        0.98 |   33 |   34 |         77 |


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
