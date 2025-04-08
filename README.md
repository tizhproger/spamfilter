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

<br/>

`Importants notice: All datasets contain ONLY english messages`\
`TF-IDF + Logistic Regression was set with parameter class_weight='balanced', in order to compensate inbalance in datasets`

<br/>

**Time** - shows how long it took a model to predict a batch of messages.

<details>
  <summary>English Datasets</summary>

### BERT Base
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9941 |            0.97 |               0.99 |        0.98 |    4 |   16 |         29 |
| Combined NoEmail (Twitter + SMS) |     0.9978 |            0.99 |               1    |        0.99 |    1 |    4 |         16 |
| SMS                              |     0.9937 |            0.97 |               0.99 |        0.98 |    2 |    5 |          8 |
| Twitter                          |     0.991  |            0.95 |               0.98 |        0.97 |    3 |    7 |          8 |
| Email                            |     0.9948 |            0.98 |               1    |        0.99 |    0 |    6 |         14 |

---

### BERT Multilingual
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9887 |            0.96 |               0.97 |        0.97 |   17 |   21 |         29 |
| Combined NoEmail (Twitter + SMS) |     0.9942 |            0.98 |               0.98 |        0.98 |    6 |    7 |         15 |
| SMS                              |     0.9946 |            0.97 |               0.99 |        0.98 |    2 |    4 |          8 |
| Twitter                          |     0.9901 |            0.96 |               0.97 |        0.96 |    5 |    6 |          8 |
| Email                            |     0.9895 |            0.97 |               0.99 |        0.98 |    3 |    9 |         14 |

---

### DeBERTa v3 Small
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9956 |            0.98 |               0.99 |        0.99 |    5 |   10 |         34 |
| Combined NoEmail (Twitter + SMS) |     0.9973 |            0.99 |               0.99 |        0.99 |    2 |    4 |         19 |
| SMS                              |     0.9937 |            0.97 |               0.99 |        0.98 |    2 |    5 |         10 |
| Twitter                          |     0.9928 |            0.97 |               0.98 |        0.97 |    3 |    5 |         10 |
| Email                            |     0.9913 |            0.97 |               1    |        0.98 |    1 |    9 |         14 |

---

### DistilBERT
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9941 |            0.98 |               0.99 |        0.98 |    7 |   13 |         16 |
| Combined NoEmail (Twitter + SMS) |     0.9991 |            0.99 |               1    |        1    |    0 |    2 |          8 |
| SMS                              |     0.9919 |            0.97 |               0.97 |        0.97 |    4 |    5 |          4 |
| Twitter                          |     0.991  |            0.95 |               0.99 |        0.97 |    2 |    8 |          4 |
| Email                            |     0.9913 |            0.98 |               0.99 |        0.98 |    3 |    7 |          8 |

---

### RoBERTa Base
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9881 |            0.96 |               0.97 |        0.96 |   19 |   21 |         30 |
| Combined NoEmail (Twitter + SMS) |     0.9906 |            0.96 |               0.97 |        0.96 |    9 |   12 |         16 |
| SMS                              |     0.9946 |            0.97 |               0.99 |        0.98 |    2 |    4 |          8 |
| Twitter                          |     0.9928 |            0.95 |               0.99 |        0.97 |    1 |    7 |          8 |
| Email                            |     0.9913 |            0.97 |               1    |        0.98 |    1 |    9 |         14 |

---

### TF-IDF + LR
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.9742 |            0.93 |               0.92 |        0.92 |   48 |   39 |          2 |
| Combined NoEmail (Twitter + SMS) |     0.9888 |            0.95 |               0.96 |        0.96 |   12 |   13 |          1 |
| SMS                              |     0.9785 |            0.93 |               0.91 |        0.92 |   13 |   11 |          1 |
| Twitter                          |     0.9812 |            0.94 |               0.92 |        0.93 |   12 |    9 |          1 |
| Email                            |     0.9895 |            0.99 |               0.97 |        0.98 |    9 |    3 |          1 |

---

### XLM-RoBERTa Base
| Dataset                          |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:---------------------------------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Combined (Email + Twitter + SMS) |     0.979  |            0.92 |               0.95 |        0.94 |   28 |   43 |         31 |
| Combined NoEmail (Twitter + SMS) |     0.9897 |            0.93 |               0.99 |        0.96 |    3 |   20 |         16 |
| SMS                              |     0.9928 |            0.97 |               0.98 |        0.97 |    3 |    5 |          8 |
| Twitter                          |     0.9857 |            0.93 |               0.97 |        0.95 |    5 |   11 |          8 |
| Email                            |     0.9887 |            0.96 |               0.99 |        0.98 |    2 |   11 |         15 |

</details>

<details>
  <summary>Russian Datasets</summary>

### BERT Multilingual
| Dataset   |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:----------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Telegram  |     0.9821 |            0.98 |               0.98 |        0.98 |   24 |   35 |         26 |

---

### RuBERT Tiny
| Dataset   |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:----------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Telegram  |     0.9821 |            0.98 |               0.98 |        0.98 |   28 |   31 |         11 |

---

### XLM-RoBERTa Base
| Dataset   |   Accuracy |   Recall (Spam) |   Precision (Spam) |   F1 (Spam) |   FP |   FN |   Time (s) |
|:----------|-----------:|----------------:|-------------------:|------------:|-----:|-----:|-----------:|
| Telegram  |     0.9746 |            0.96 |               0.99 |        0.97 |   23 |   61 |         25 |

</details>

## Visualizations

### Metric Comparison

![Metric Comparison](metrics_comparison.png)

![Metric Comparison](metrics_comparison.png)

![Metric Comparison](metrics_comparison.png)

## Findings

- **Email data introduces noise**, increasing false negatives and lowering recall.
- **TF-IDF** is more sensitive to structured noise (e.g., headers, HTML) in email content, but still effective.
- **DeBERTa** consistently delivers high performance across datasets, achieving near-perfect results when trained without noisy email data.

---

## Recommendation

- For simplicity and reproducibility, **TF-IDF + LogisticRegression** remains a solid option.
- For **highest accuracy and robustness**, especially across diverse text formats, **DeBERTa is the better choice**.
