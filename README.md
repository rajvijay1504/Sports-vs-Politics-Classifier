# Sports vs. Politics Document Classifier
**Author:** Raj Vijayvargiya  
**Roll Number:** B22AI067  
**Course:** B.Tech Final Year, IIT Jodhpur

## Project Overview
This project implements a complete machine learning pipeline to classify text documents into two categories: **Sports** and **Politics**. The goal was to compare different mathematical approaches to text classification—probabilistic, linear, and margin-based—to identify the most robust model for newsgroup data.

## Dataset
The project uses the **20 Newsgroups dataset**, specifically filtered for binary classification:
- **Sports:** `rec.sport.baseball`, `rec.sport.hockey`
- **Politics:** `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`

**Dataset Statistics:**
- Total Samples: 4,618
- Politics: 2,625 samples
- Sports: 1,993 samples

## Methodology
1. **Preprocessing:** Stripping headers, footers, and quotes to ensure the model learns from core content rather than metadata.
2. **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert text into numerical vectors, highlighting domain-specific "signature words".
3. **Algorithms Compared:**
   - **Multinomial Naive Bayes:** A probabilistic approach.
   - **Logistic Regression:** A linear approach.
   - **Linear SVM:** A margin-based approach.

## Quantitative Results
The models were evaluated using an 80-20 train-test split (899 test samples).

| **Model** | **Accuracy** |
| :--- | :--- |
| Multinomial Naive Bayes | 96.55% |
| Linear SVM | 95.22% |
| Logistic Regression | 94.99% |

*Data source: 20 Newsgroups Dataset via Scikit-Learn*

## Key Findings
- **Naive Bayes** achieved the highest accuracy, excelling because the vocabularies for Sports and Politics are highly distinct.
- **Feature Importance:** TF-IDF analysis showed that words like "game," "team," and "hockey" were the strongest predictors for Sports, while "government," "israel," and "gun" were primary for Politics.
- **Precision vs. Recall:** The system achieved a Sports precision of 0.98, meaning it is exceptionally accurate when identifying sports-related content.

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install scikit-learn pandas matplotlib seaborn

```

3. Run the script:
```bash
python B22AI067_prob4.py

```


