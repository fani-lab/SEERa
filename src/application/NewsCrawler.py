import pandas as pd
import numpy as np
from newspaper import Article
import re

#tweetentities_path = '../../data/tweetentities.csv'
tweetentities_path = '../../data/main_tables_csv/TweetEntities.csv'

tweetentities_table = pd.read_csv(tweetentities_path, delimiter=';')
tweetentities_table.dropna(inplace=True, subset=['ExpandedUrl'])
URLs = tweetentities_table['ExpandedUrl']
ShortURLs = tweetentities_table['Url']
DisplayURLs = tweetentities_table['DisplayUrl']
TweetIds = tweetentities_table['TweetId']
print(URLs.shape)
print(TweetIds.shape)

newsArticles = []
newsTitles = []
TwIds = []
publishDate = []
accepted_URLs = []
accepted_ShortURLs = []
accepted_DisplayURLs = []
description = []
source_URLs = []
for count, url in zip(URLs.index, URLs.values):
    if count % 500 == 0:
        print(count, ' / ', len(URLs))
    #if count == 100:
    #    break
    try:
        if url in accepted_URLs:
            continue
        article = Article(url)
        article.download()
        article.parse()
        #TwIds.append(TweetIds[count])

        accepted_ShortURLs.append(ShortURLs[count])
        accepted_DisplayURLs.append(DisplayURLs[count])
        accepted_URLs.append(url)
        source_URLs.append(article.source_url)
        text = article.text
        #print(text)
        # text = text.replace("'", "")
        title = article.title
        #print(title)
        # title = title.replace("'", "")
        publishDate.append(article.publish_date)
        #print(publishDate)
        #print('publish date:', publishDate)
        newsArticles.append(text)
        newsTitles.append(title)
        description.append(article.meta_description)
        #boogh += 1
        #print('------------------------------')
    except:
        pass

News = {'ExpandedUrl': accepted_URLs, 'ShortUrl': accepted_ShortURLs, 'DisplayUrl': accepted_DisplayURLs,
        'SourceUrl': source_URLs, 'Text': newsArticles, 'Title': newsTitles, 'Description': description,
        'PublicationTime': publishDate}

News = pd.DataFrame.from_dict(News)
News.to_csv('../../data/main_tables_csv/NewNews.csv', index=False)
print('CSV file saved.')