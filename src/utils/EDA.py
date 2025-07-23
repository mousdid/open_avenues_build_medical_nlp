#imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import io

def load_and_preview_dataset(filepath):
    """
    Load the dataset and return the DataFrame, its shape, and the first 15 rows.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        tuple: (DataFrame, shape tuple, DataFrame head)
    """
    reports = pd.read_csv(filepath)
    shape = reports.shape
    head = reports.head(15)
    return reports, shape, head





def text_length_insights(reports):
    """
    Adds character count, word count, and average word length columns to the DataFrame,
    and returns the figures for use in Metaflow cards.
    Args:
        reports (pd.DataFrame): DataFrame with 'ReportText' column.
    Returns:
        pd.DataFrame: Updated DataFrame.
        tuple: (fig_char_count, fig_word_count, fig_avg_word_length)
    """
    # Character count
    reports['char_count'] = reports['ReportText'].str.len()
    # Word count
    reports['word_count'] = reports['ReportText'].str.split().map(len)
    # Average word length
    reports['avg_word_length'] = reports['ReportText'].str.split().apply(
        lambda x: np.mean([len(i) for i in x]) if x else 0
    )

    # Character count figure
    fig_char_count, ax1 = plt.subplots()
    reports['char_count'].hist(ax=ax1, bins=50, color='skyblue')
    ax1.set_title('Character Count Distribution')
    ax1.set_xlabel('Characters')

    # Word count figure
    fig_word_count, ax2 = plt.subplots()
    reports['word_count'].hist(ax=ax2, bins=50, color='lightgreen')
    ax2.set_title('Word Count Distribution')
    ax2.set_xlabel('Words')

    # Average word length figure
    fig_avg_word_length, ax3 = plt.subplots()
    reports['avg_word_length'].hist(ax=ax3, bins=30, color='salmon')
    ax3.set_title('Average Word Length Distribution')
    ax3.set_xlabel('Avg Word Length')

    return (fig_char_count, fig_word_count, fig_avg_word_length)

def most_frequent_stopwords(reports):
    """
    Finds and plots the most frequent stopwords in the ReportText column.
    Args:
        reports (pd.DataFrame): DataFrame with 'ReportText' column.
    Returns:
        tuple: (list of top stopwords and their counts, matplotlib figure)
    """


    stop = set(stopwords.words('english'))
    report = reports['ReportText'].str.split()
    corpus = [word for words in report for word in words]

    dic = defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word] += 1

    top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]
    x, y = zip(*top)

    fig, ax = plt.subplots()
    ax.bar(x, y, color='orchid')
    ax.set_title('Top 10 Stopwords')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Stopword')

    return top, fig

def most_frequent_words(reports, top_n=40):
    """
    Finds and plots the most frequent stopwords and non-stopwords in the ReportText column.
    Args:
        reports (pd.DataFrame): DataFrame with 'ReportText' column.
        top_n (int): Number of top words to return for non-stopwords.
    Returns:
        dict: {'stopwords': list of (word, count), 'non_stopwords': list of (word, count)}
        tuple: (stopwords bar figure, non-stopwords bar figure)
    """


    stop = set(stopwords.words('english'))
    report = reports['ReportText'].str.split()
    corpus = [word for words in report for word in words]

    # Stopwords
    dic = defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word] += 1
    top_stop = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]
    x_stop, y_stop = zip(*top_stop)
    fig_stop, ax_stop = plt.subplots()
    ax_stop.bar(x_stop, y_stop, color='orchid')
    ax_stop.set_title('Top 10 Stopwords')
    ax_stop.set_ylabel('Frequency')
    ax_stop.set_xlabel('Stopword')

    # Non-stopwords
    counter = Counter(corpus)
    most = counter.most_common()
    x_nonstop, y_nonstop = [], []
    for word, count in most[:top_n]:
        if word not in stop:
            x_nonstop.append(word)
            y_nonstop.append(count)
    fig_nonstop, ax_nonstop = plt.subplots(figsize=(8, 10))
    sns.barplot(x=y_nonstop, y=x_nonstop, ax=ax_nonstop, palette='Blues_r')
    ax_nonstop.set_title(f'Top {top_n} Non-Stopwords')
    ax_nonstop.set_xlabel('Frequency')
    ax_nonstop.set_ylabel('Word')

    return {'stopwords': top_stop, 'non_stopwords': list(zip(x_nonstop, y_nonstop))}, (fig_stop, fig_nonstop)

def get_top_ngrams(reports, n=2, top_k=10):
    """
    Finds and plots the most frequent n-grams (bigrams or trigrams) in the ReportText column.
    Args:
        reports (pd.DataFrame): DataFrame with 'ReportText' column.
        n (int): n-gram size (2 for bigrams, 3 for trigrams).
        top_k (int): Number of top n-grams to return.
    Returns:
        list: Top n-grams and their counts.
        matplotlib.figure.Figure: Barplot figure for Metaflow card.
    """


    corpus = reports['ReportText'].astype(str).tolist()
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]

    x, y = zip(*words_freq)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=y, y=x, ax=ax, palette='viridis')
    ax.set_title(f'Top {top_k} {n}-grams')
    ax.set_xlabel('Frequency')
    ax.set_ylabel(f'{n}-gram')

    return words_freq, fig

def show_wordcloud(reports):
    """
    Generates a word cloud from the ReportText column.
    Args:
        reports (pd.DataFrame): DataFrame with 'ReportText' column.
    Returns:
        matplotlib.figure.Figure: Word cloud figure for Metaflow card.
    """
    stopwords_set = set(STOPWORDS)
    text = " ".join(reports['ReportText'].astype(str).tolist())
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords_set,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1
    ).generate(text)

    fig = plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(wordcloud)
    return fig

def preprocessing(reports):
    """
    Removes rows with missing values from the DataFrame.
    Args:
        reports (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: Cleaned DataFrame with missing rows dropped.
        int: Number of rows dropped.
    """
    initial_count = len(reports)
    cleaned_reports = reports.dropna()
    dropped_count = initial_count - len(cleaned_reports)
    return cleaned_reports, dropped_count










def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.read()



