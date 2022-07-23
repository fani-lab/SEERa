import pandas as pd
import numpy as np
from newspaper import Article
from tqdm import tqdm
import glob

def CrawlRequest(url):
    article = Article(url)
    article.download()
    article.parse()
    return article

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
            description.append(article.meta_description)
        except:
            pass
    if not chunk:
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

