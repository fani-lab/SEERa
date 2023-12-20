import pandas as pd
from newspaper import Article

def main(documents):
    # Create a DataFrame for extracted information
    extracted_info = pd.DataFrame(
        columns=['linkId', 'URL', 'title', 'text', 'description', 'publish date'])
    visited_urls = []
    id_index = 0
    missing_links = 0
    linkIDs = []
    for index, row in documents.iterrows():
        linkIDs.append([])
        for url in row['Extracted_Links']:
            if url in visited_urls:
                linkIDs[-1].append(visited_urls.index(url))
                continue
            article = Article(url)
            print(url)
            try:
                article.download()
                article.parse()

                title = article.title
                text = article.text
                description = article.meta_description
                publish_date = article.publish_date

                extracted_info = extracted_info.append({
                    'linkId': id_index,
                    'URL': url,
                    'title': title,
                    'text': text,
                    'description': description,
                    'publish date': publish_date
                }, ignore_index=True)
                linkIDs[-1].append(id_index)
                id_index += 1

            except:
                missing_links += 1
    print('missing_links:', missing_links)
    # Print the extracted information DataFrame
    documents['URL_ids'] = linkIDs
    extracted_info.to_csv('NewsFromTweets.csv')
    # linkIDs = []
    # for index, row in documents.iterrows():
    #     linkIDs.append([])
    #     for url in row['Extracted_Links']:
    #         linkIDs[-1].append()
    print(extracted_info)
    return documents, extracted_info