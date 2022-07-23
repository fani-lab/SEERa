import pandas as pd
<<<<<<< HEAD
import os
from newspaper import Article
import params
from tqdm import tqdm
import glob

def crawl_request(url):
=======
import numpy as np
from newspaper import Article
from tqdm import tqdm
import glob

def CrawlRequest(url):
>>>>>>> 54bc47755b58cb9863c9bd516cbac720613f723c
    article = Article(url)
    article.download()
    article.parse()
    return article

<<<<<<< HEAD

def news_crawler(path):
    tweet_entities = pd.read_csv(path)
    tweet_entities.dropna(inplace=True, subset=['ExpandedUrl'])
    urls = tweet_entities['ExpandedUrl']
    short_urls = tweet_entities['Url']
    display_urls = tweet_entities['DisplayUrl']
    tweet_ids = tweet_entities['TweetId']
    entity_type_codes = tweet_entities['EntityTypeCode']
    news_articles = []
    news_titles = []
    publish_date = []
    accepted_urls = []
    accepted_short_urls = []
    accepted_display_urls = []
    description = []
    source_urls = []
    chunk = True
    if chunk and not os.path.isdir(f'../data/toy/News'): os.makedirs(f'../data/toy/News')
    chunk_size = 20000
    indices = urls.index
    url_values = urls.values
    for i in tqdm(range(len(url_values))):
        if entity_type_codes[i] != 2:
            continue
        url = url_values[i]
        ind = indices[i]
        if chunk and i % chunk_size == 0 and i > 0:
            news = {'ExpandedUrl': accepted_urls, 'ShortUrl': accepted_short_urls, 'DisplayUrl': accepted_display_urls,
                    'SourceUrl': source_urls, 'Text': news_articles, 'Title': news_titles, 'Description': description,
                    'PublicationTime': publish_date}
            news = pd.DataFrame.from_dict(news)
            news.to_csv(f'../data/toy/News/News_Chunk{i//chunk_size}.csv', index=False)
            accepted_urls = []
            accepted_short_urls = []
            accepted_display_urls = []
            description = []
            source_urls = []
            news_articles = []
            news_titles = []
            publish_date = []
        try:
            if url in accepted_urls:
                continue
            article = crawl_request(url)
            accepted_short_urls.append(short_urls[ind])
            accepted_display_urls.append(display_urls[ind])
            accepted_urls.append(url)
            source_urls.append(article.source_url)
            text = article.text
            title = article.title
            publish_date.append(article.publish_date)
            news_articles.append(text)
            news_titles.append(title)
=======
def NewsCrawler(path):
    tweetentities_path = path
    tweetentities_table = pd.read_csv(tweetentities_path)
    tweetentities_table.dropna(inplace=True, subset=['ExpandedUrl'])
    URLs = tweetentities_table['ExpandedUrl']
    ShortURLs = tweetentities_table['Url']
    DisplayURLs = tweetentities_table['DisplayUrl']
    TweetIds = tweetentities_table['TweetId']
    EntityTypeCodes = tweetentities_table['EntityTypeCode']
    newsArticles = []
    newsTitles = []
    TwIds = []
    publishDate = []
    accepted_URLs = []
    accepted_ShortURLs = []
    accepted_DisplayURLs = []
    description = []
    source_URLs = []
    chunk = True
    chunk_size = 20000
    indices = URLs.index
    URL_VALUES = URLs.values
    for i in tqdm(range(len(URL_VALUES))):
        if EntityTypeCodes[i] != 2:
            continue
        url = URL_VALUES[i]
        ind = indices[i]
        if chunk and i % chunk_size == 0 and i > 0:
            News = {'ExpandedUrl': accepted_URLs, 'ShortUrl': accepted_ShortURLs, 'DisplayUrl': accepted_DisplayURLs,
                    'SourceUrl': source_URLs, 'Text': newsArticles, 'Title': newsTitles, 'Description': description,
                    'PublicationTime': publishDate}
            News = pd.DataFrame.from_dict(News)
            News.to_csv(f'../../data/toy/NewNewsOO/NewNews_Chunk{i//chunk_size}.csv', index=False)
            accepted_URLs = []
            accepted_ShortURLs = []
            accepted_DisplayURLs = []
            description = []
            source_URLs = []
            newsArticles = []
            newsTitles = []
            publishDate = []
        try:
            if url in accepted_URLs:
                continue
            article = CrawlRequest(url)
            accepted_ShortURLs.append(ShortURLs[ind])
            accepted_DisplayURLs.append(DisplayURLs[ind])
            accepted_URLs.append(url)
            source_URLs.append(article.source_url)
            text = article.text
            title = article.title
            publishDate.append(article.publish_date)
            newsArticles.append(text)
            newsTitles.append(title)
>>>>>>> 54bc47755b58cb9863c9bd516cbac720613f723c
            description.append(article.meta_description)
        except:
            pass
    if not chunk:
<<<<<<< HEAD
        news = {'ExpandedUrl': accepted_urls, 'ShortUrl': accepted_short_urls, 'DisplayUrl': accepted_display_urls,
                'SourceUrl': source_urls, 'Text': news_articles, 'Title': news_titles, 'Description': description,
                'PublicationTime': publish_date}
        news = pd.DataFrame.from_dict(news)
        news.to_csv(f'../data/toy/News.csv', index=False)
    else:
        frame_path = sorted(glob.glob(f'../data/toy/News/*_Chunk*.csv'))
        frames = []
        for f in frame_path:
            frames.append(pd.read_csv(f))
        news = pd.concat(frames, ignore_index=True)
        news.to_csv(f'../data/toy/News.csv', index=False)
=======
        News = {'ExpandedUrl': accepted_URLs, 'ShortUrl': accepted_ShortURLs, 'DisplayUrl': accepted_DisplayURLs,
                'SourceUrl': source_URLs, 'Text': newsArticles, 'Title': newsTitles, 'Description': description,
                'PublicationTime': publishDate}
        News = pd.DataFrame.from_dict(News)
        News.to_csv(f'../../data/toy/News.csv', index=False)
    else:
        frame_path = sorted(glob.glob("f'../../data/toy/NewNewsOO/*_Chunk*.csv"))
        frames = []
        for f in frame_path:
            frames.append(pd.read_csv(f))
        News = pd.concat(frames, ignore_index=True)
        News.to_csv('News.csv', index=False)
>>>>>>> 54bc47755b58cb9863c9bd516cbac720613f723c

