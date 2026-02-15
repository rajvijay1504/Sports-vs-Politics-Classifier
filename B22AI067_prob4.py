import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# step 1: data collection & preparation
def collect_data():
    # we use the 20 newsgroups dataset to get clean text data
    print("Step 1: Collecting Data from 20 Newsgroups Dataset")
    # choosing specific labels for sports and politics
    categories=['rec.sport.baseball', 'rec.sport.hockey', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc']
    
    # downloading the dataset content
    # we strip headers/footers to avoid model cheating on email signatures
    print("Loading dataset...")
    dataset=fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    # building the dataframe
    df=pd.DataFrame({'text': dataset.data, 'target': dataset.target})
    df['label_name']=df['target'].apply(lambda x: dataset.target_names[x])
    
    # categorizing into binary groups
    df['category']=df['label_name'].apply(lambda x: 'Sports' if 'sport' in x else 'Politics')
    
    print(f"Data Collected! Total samples: {len(df)}")
    print(df['category'].value_counts())
    return df

# new function for the 4th png: vocab visualization
def visualize_vocab(tfidf, X, y):
    # this helps us see which words the model thinks are important
    print("Step 2.1: Generating Vocabulary Visualizations")
    feature_names=tfidf.get_feature_names_out()
    
    # we convert to a dense array to calculate the average weights
    dense_x=X.todense()
    df_tfidf=pd.DataFrame(dense_x, columns=feature_names)
    df_tfidf['class_label']=y.values
    
    # calculating the average importance per category
    mean_tfidf=df_tfidf.groupby('class_label').mean()
    
    for category in ['Sports', 'Politics']:
        # grabbing the top 10 words for each group
        top_words=mean_tfidf.loc[category].sort_values(ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_words.values, y=top_words.index, palette='magma')
        plt.title(f'Top 10 Keywords for {category} by TF-IDF Weight')
        plt.xlabel('Average TF-IDF Score')
        plt.ylabel('Token')
        # this will pop up a window you can save as a PNG
        plt.show()

# step 2: feature extraction (tfidf)
def extract_features(df):
    print("\nStep 2: Extracting TF-IDF Features")
    # transforming raw text into numeric tf-idf vectors
    # we limit to 5000 features to avoid overfitting
    tfidf=TfidfVectorizer(stop_words='english', max_features=5000)
    X=tfidf.fit_transform(df['text'])
    y=df['category']
    
    # calling our new visualizer here
    visualize_vocab(tfidf, X, y)
    
    print(f"Feature Matrix Shape: {X.shape}")
    return X, y

# step 3: model training & comparison
def compare_models(X, y):
    print("\nStep 3: Training and Comparing Models")
    # splitting data 80-20
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
    
    # defining the three ml techniques to compare
    models={
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(random_state=42, dual='auto')
    }
    
    results={}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        
        acc=accuracy_score(y_test, y_pred)
        results[name]=acc
        
        print(f"\nRESULTS FOR {name.upper()}:")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # generating confusion matrices for the report
        cm=confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Politics', 'Sports'], yticklabels=['Politics', 'Sports'])
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()

    return results

if __name__ == "__main__":
    # execute the whole pipeline
    df=collect_data()
    
    # clearing out any rows that are actually empty
    df.dropna(subset=['text'], inplace=True)
    df=df[df['text'].str.strip().astype(bool)]
    
    X, y=extract_features(df)
    results=compare_models(X, y)
    
    print("\nFinal Comparison Summary")
    for model_name, acc in results.items():
        print(f"{model_name}: {acc:.4f}")