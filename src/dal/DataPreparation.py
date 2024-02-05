import datetime

import pandas as pd
from cmn import Common as cmn
import params
import re


def stat(documents):
    import matplotlib.pyplot as plt
    from collections import Counter
    tokens_column = documents['Tokens']
    # Flatten the list of lists into a single list of tokens
    all_tokens = [token for tokens_list in tokens_column for token in tokens_list]
    # Count the occurrences of each word
    word_counts = Counter(all_tokens)
    # Find the 50 most frequent words
    most_common_words = word_counts.most_common(50)
    # Plot the word distribution
    words, counts = zip(*most_common_words)
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 50 Most Frequent Words')
    plt.xticks(rotation='vertical')
    plt.savefig(f"{params.dal['statPath2Save']}/TopFrequentWords.png")
    plt.close()
    # Save the word distribution to a file
    with open(f"{params.dal['statPath2Save']}/word_distribution.txt", 'w') as file:
        for word, count in most_common_words:
            file.write(f"{word}: {count}\n")
    # Calculate token frequencies
    token_frequencies = list(word_counts.values())
    max_frequency = max(token_frequencies)
    num_bins = 10
    bin_size = max_frequency // num_bins
    # Create bins for the histogram
    bins = [i * bin_size for i in range(1, num_bins + 1)]

    # Plot the histogram
    plt.hist(token_frequencies, bins=bins, edgecolor='black')
    plt.xlabel('Token Frequency')
    plt.ylabel('Number of Tokens')
    plt.xticks(rotation='vertical')
    plt.title('Distribution of Token Frequencies')
    for bin_start, bin_end in zip(bins[:-1], bins[1:]):
        count_in_range = len([freq for freq in token_frequencies if bin_start <= freq <= bin_end])
        plt.text(bin_start + bin_size / 2, count_in_range, str(count_in_range), ha='center', va='bottom')

    # Save the histogram plot
    plt.savefig(f"{params.dal['statPath2Save']}/token_frequency_distribution.png")
    plt.close()
    # Save the result to a text file
    with open(f"{params.dal['statPath2Save']}/token_frequency_result.txt", 'w') as file:
        file.write("Token Frequency Distribution:\n")
        for bin_start, bin_end in zip(bins[:-1], bins[1:]):
            count_in_range = len([freq for freq in token_frequencies if bin_start <= freq < bin_end])
            file.write(f"{bin_start}-{bin_end - 1}: {count_in_range} tokens\n")
def reassign_id(table, column):
    posts = table[column].unique()
    new_ids = list(range(len(table[column].unique())))
    mapping = dict(zip(posts, new_ids))
    table[column] = table[column].map(mapping)
    return table

def date2timestamp(table,date_col):
    date_time_obj = datetime.datetime.strptime(params.dal['start'], '%Y-%m-%d').date()
    startDateOrdinal = date_time_obj.toordinal()
    timeStamps = []
    for index, row in table.iterrows():
        dayDiff = row[date_col].toordinal() - startDateOrdinal
        timeStamps.append((dayDiff // params.dal['timeInterval']))
    table['TimeStamp'] = timeStamps
    return table

def preprocess_tweets(text):
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt')
    nltk.download('stopwords')
    preprocessed_text = text.apply(lambda x: x.lower())
    preprocessed_text = preprocessed_text.apply(lambda x: re.sub(r'@\w+', '', x))
    preprocessed_text = preprocessed_text.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    preprocessed_text = preprocessed_text.apply(word_tokenize)
    stop_words = set(stopwords.words('english'))
    gist_file = open(params.dal['stopwordPath'], "r")
    try:
        content = gist_file.read()
        stopwords2 = content.split(",")
    finally:
        gist_file.close()
    preprocessed_text = preprocessed_text.apply(lambda tokens: [token for token in tokens if token not in stop_words and token not in stopwords2 and 2 < len(token) <= 10])
    return pd.Series(preprocessed_text)


def data_preparation(dataset):
    cols_to_drop = ['ModificationTimestamp']
    dataset.dropna(inplace=True)
    dataset.drop(cols_to_drop, axis=1, inplace=True)
    dataset = dataset.sort_values(by="CreationDate")
    dataset = date2timestamp(dataset, 'CreationDate')
    cmn.logger.info(f'DataPreperation: userModeling={params.dal["userModeling"]}, timeModeling={params.dal["timeModeling"]}, preProcessing={params.dal["preProcessing"]}, TagME={params.dal["tagMe"]}')
    if params.dal['userModeling'] and params.dal['timeModeling']:
        documents = dataset.groupby(['UserId', 'TimeStamp']).agg({'Text': lambda x: ' '.join(x), 'Extracted_Links': 'sum'}).reset_index()
    elif params.dal['userModeling']:
        documents = dataset.groupby(['UserId']).agg({'Text': lambda x: ' '.join(x), 'Extracted_Links': 'sum'}).reset_index()
    elif params.dal['timeModeling']:
        documents = dataset.groupby(['TimeStamp']).agg({'Text': lambda x: ' '.join(x), 'Extracted_Links': 'sum'}).reset_index()
    else:
        documents = dataset
    if params.dal['preProcessing']: documents['Tokens'] = preprocess_tweets(documents['Text'])
    else: documents['Tokens'] = documents['Text'].str.split()
    documents = documents[documents['Tokens'].apply(lambda x: len(x)>0)].reset_index()

    # Assuming documents is your DataFrame
    max_timestamp = documents['TimeStamp'].max()

    # Create an empty list to store DataFrames for each user
    user_dfs = []

    # Iterate over unique UserIds
    for user_id in documents['UserId'].unique():
        # Filter rows for the current user
        user_df = documents[documents['UserId'] == user_id]
        user_dfs.append(user_df)
        # Check if the user has rows for all timestamps
        # if set(user_df['TimeStamp']) == set(range(max_timestamp + 1)):
        #     # Append the DataFrame to the list
        #     user_dfs.append(user_df)

    # Concatenate DataFrames for users who have rows for all timestamps
    documents = pd.concat(user_dfs, ignore_index=True)



    if params.dal['getStat']: stat(documents)
    documents.to_csv(f"../output/{params.general['baseline']}/documents.csv", encoding='utf-8', index=False,header=True)
    cmn.logger.info(f'DataPreparation: Documents shape: {documents.shape}')
    return documents