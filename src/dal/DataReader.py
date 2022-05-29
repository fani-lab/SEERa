import sys
import pandas as pd
# sys.path.extend(["../"])

def GoldenStandard2Reader():
    table = pd.read_csv(r'../../data/goldenstandard.csv', sep=';',
                        encoding='utf-8') #loads all from GoldenStandard CSV into dataframe
    return table


def load_tweets(path, startDate, endDate, stopwords=['www', 'RT', 'com', 'http'], tagme_threshold=0.02):
    tweets_path, tagmeannotation_path = path
    # reading csv file and creating dataframe with name tweets, change as necessary
    tweets = pd.read_csv(tweets_path, sep=';', encoding='utf-8')
    #print('tw1',tweets)
    # changing name of Id to TweetId to allow merge with tagme dataframe
    tweets.rename(columns={'Id': 'TweetId'}, inplace=True)
    #tweets = tweets.drop(columns=["Text"])  # drop unwanted columns named Text
    tweets = tweets[tweets.TweetId != -1]  # remove rows with tweet ids -1 value
    tweets = tweets[tweets.UserId != -1]  # remove rows with user ids with -1 value

    tagme = pd.read_csv(tagmeannotation_path, sep=',', encoding='utf-8')  # creating tagme dataframe from csv
    #print('tg1',tagme)
    tagme.Word = tagme.Word.astype(str)
    tagme = tagme[tagme.Score > tagme_threshold]  # drop scores below .07
    tagme = tagme[~tagme['Word'].isin(stopwords)]  # drop rows with stopwords
    tagme = tagme.drop(columns=["StartIndex", "EndIndex", "ConceptId", "IsDeleted", "Id",
                                "Score"])  # drop unwanted columns
    tagme.Word = tagme.Word.astype('string')  # declare type of word as string
    #print('tg2', tagme)
    tagme = tagme.groupby('TweetId').agg(
        lambda word: ','.join(word))#.tolist())  # take all words with same TweetId and put them in a list

    tweets = pd.merge(tweets, tagme, on='TweetId')  # merge dataframes
    #print('tw2', tweets)
    tweets.rename(columns={"CreationTimestamp": "CreationDate", "Word": "Tokens"},
                  inplace=True)  # rename columns to match sql query

    tweets['ModificationTimestamp'] = pd.to_datetime(tweets['ModificationTimestamp'])
    tweets['CreationDate'] = pd.to_datetime(tweets['CreationDate']) # declare that creationdate and modification timestamp is a datetime value

    # locate only the tweets we want to actually grab and save to tweets dataframe from the entered start and end date
    tweets = tweets.loc[(tweets['CreationDate'] >= startDate)
                        & (tweets['CreationDate'] < endDate)]

    # print(tweets.to_string()) # for viewing table during testing
    #print('tw3', tweets)
    return tweets




## test
#path= ['../../data/tweets_N20000.csv', '../../data/tagmeannotation_N80000.csv']
#a = load_tweets(path=path, startDate='2000-10-06', endDate='2021-01-01', stopwords=['www', 'RT', 'com', 'http'], tagme_threshold=0.07)

































