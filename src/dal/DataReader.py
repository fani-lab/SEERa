import os, re, datetime
import pandas as pd

from cmn import Common as cmn
import params


def extract_link(text):
    matches = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return matches if matches else []
def load_tweets(path):
    if not os.path.isdir(params.dal['statPath2Save']): os.makedirs(params.dal['statPath2Save'])
    start = datetime.datetime.strptime(params.dal['start'], '%Y-%m-%d')
    end = datetime.datetime.strptime(params.dal['end'], '%Y-%m-%d')
    # Leaving params.dal['testTimeIntervals'] intervals for testing
    end = end - datetime.timedelta(days=params.dal["testTimeIntervals"]*params.dal["timeInterval"])

    tweets = pd.read_csv(path, encoding='utf-8', parse_dates=['CreationTimestamp'])

    # picking 5% of the data to make the process faster
    tweets = tweets.sample(frac=0.05, random_state=42)


    tweets.rename(columns={'Id': 'TweetId', 'CreationTimestamp': 'CreationDate', 'Tokens': 'Text'}, inplace=True)
    tweets = tweets[(tweets.TweetId != -1) & (tweets.UserId != -1)]  # remove rows with tweet ids -1 value or user ids with -1 value
    # only keep tweets between start date and end date
    tweets = tweets.loc[(tweets['CreationDate'].dt.date >= start.date()) & (tweets['CreationDate'].dt.date <= end.date())]

    # declare that creationdate and modification timestamp are datetime values
    tweets['CreationDate'] = pd.to_datetime(tweets['CreationDate'])
    if params.dal['addLinks']:
        links = tweets['Text'].apply(extract_link)
        tweets['Extracted_Links'] = links
    if params.dal['wantStats']:
        stats = stat(tweets)
        print(stats)
    return tweets

def stat(tweets):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Basic statistics
    num_tweets = tweets['TweetId'].nunique()
    num_users = tweets['UserId'].nunique()

    # Function to check if a text contains a link
    def has_link(text):
        return bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))

    # Extract links from tweets
    links = tweets['Text'].apply(extract_link)
    tweets['Extracted_Links'] = links

    # Calculate stats related to links
    num_tweets_with_links = tweets['Extracted_Links'].apply(lambda x: len(x) > 0).sum()
    unique_users_with_links = tweets[tweets['Extracted_Links'].apply(len) > 0]['UserId'].nunique()
    unique_users_without_links = num_users - unique_users_with_links

    # Count the number of tweets per user
    tweets_per_user = tweets['UserId'].value_counts()

    # Count the number of users for each count of tweets
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800,
            900, 1000, float('inf')]
    user_counts = pd.cut(tweets_per_user, bins=bins).value_counts().sort_index()

    # Additional statistics related to links
    total_links = tweets['Extracted_Links'].apply(len).sum()
    links_per_user = tweets.groupby('UserId')['Extracted_Links'].apply(lambda x: x.apply(len).sum()).fillna(0)

    # Additional statistics
    avg_tweets_per_user = tweets_per_user.mean()
    max_tweets_by_user = tweets_per_user.max()
    min_tweets_by_user = tweets_per_user.min()

    # Log statistics during runtime
    cmn.logger.info(f'Number of tweets: {num_tweets}')
    cmn.logger.info(f'Number of users: {num_users}')
    cmn.logger.info(f'Number of tweets with links: {num_tweets_with_links}')
    cmn.logger.info(f'Number of users with links: {unique_users_with_links}')
    cmn.logger.info(f'Number of users without links: {unique_users_without_links}')
    cmn.logger.info(f'Total number of links: {total_links}')
    cmn.logger.info(f'Average tweets per user: {avg_tweets_per_user}')
    cmn.logger.info(f'Maximum tweets by a user: {max_tweets_by_user}')
    cmn.logger.info(f'Minimum tweets by a user: {min_tweets_by_user}')

    # Plots
    # 1. Number of tweets per user
    plt.figure(figsize=(10, 6))
    sns.barplot(x=tweets_per_user.index, y=tweets_per_user.values, color='skyblue', edgecolor='black')
    plt.xlabel('User ID')
    plt.ylabel('Number of Tweets')
    plt.title('Number of Tweets per User')
    plt.savefig(f'{params.dal["statPath2Save"]}/tweets_per_user.png')
    plt.close()

    # 2. Distribution of tweet counts per user
    plt.figure(figsize=(10, 6))
    sns.histplot(user_counts, bins=20, color='salmon', edgecolor='black')
    plt.xlabel('Number of Tweets')
    plt.ylabel('Number of Users')
    plt.title('Number of Users for Each Count of Tweets')
    plt.savefig(f'{params.dal["statPath2Save"]}/user_counts_distribution.png')
    plt.close()

    # 3. Proportion of tweets with links
    plt.figure(figsize=(8, 8))
    plt.pie([num_tweets_with_links, num_tweets - num_tweets_with_links], labels=['With Links', 'Without Links'],
            autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
    plt.title('Proportion of Tweets with Links')
    plt.savefig(f'{params.dal["statPath2Save"]}/proportion_tweets_with_links.png')
    plt.close()

    # 4. Total links per user
    plt.figure(figsize=(10, 6))
    sns.barplot(x=links_per_user.index, y=links_per_user.values, color='lightgreen', edgecolor='black')
    plt.xlabel('User ID')
    plt.ylabel('Total Links')
    plt.title('Total Links per User')
    plt.savefig(f'{params.dal["statPath2Save"]}/total_links_per_user.png')
    plt.close()

    # 5. Average tweets per user
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=tweets_per_user.values, color='lightcoral')
    plt.ylabel('Number of Tweets')
    plt.title('Boxplot of Tweets per User')
    plt.savefig(f'{params.dal["statPath2Save"]}/boxplot_tweets_per_user.png')
    plt.close()

    # 6. Distribution of link counts per tweet
    plt.figure(figsize=(10, 6))
    sns.histplot(tweets['Extracted_Links'].apply(len), bins=range(0, tweets['Extracted_Links'].apply(len).max() + 1), color='skyblue', edgecolor='black')
    plt.xlabel('Number of Links')
    plt.ylabel('Number of Tweets')
    plt.title('Distribution of Link Counts per Tweet')
    plt.savefig(f'{params.dal["statPath2Save"]}/distribution_link_counts_per_tweet.png')
    plt.close()

    # 7. Number of users with and without links
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['With Links', 'Without Links'], y=[unique_users_with_links, unique_users_without_links],
                palette=['lightblue', 'lightcoral'])
    plt.xlabel('Link Presence')
    plt.ylabel('Number of Users')
    plt.title('Number of Users with and Without Links')
    plt.savefig(f'{params.dal["statPath2Save"]}/users_with_and_without_links.png')
    plt.close()

    # 8. Distribution of tweets with links
    plt.figure(figsize=(10, 6))
    sns.histplot(tweets_per_user[tweets_per_user > 0], bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel('Number of Tweets with Links')
    plt.ylabel('Number of Users')
    plt.title('Distribution of Users with Links')
    plt.savefig(f'{params.dal["statPath2Save"]}/distribution_users_with_links.png')
    plt.close()

    # 9. Distribution of total links
    plt.figure(figsize=(10, 6))
    sns.histplot(links_per_user, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Total Links per User')
    plt.ylabel('Number of Users')
    plt.title('Distribution of Users Based on Total Links')
    plt.savefig(f'{params.dal["statPath2Save"]}/distribution_users_total_links.png')
    plt.close()

    # 10. Proportion of link counts per tweet
    plt.figure(figsize=(8, 8))
    plt.pie(tweets['Extracted_Links'].apply(len).value_counts(), labels=tweets['Extracted_Links'].apply(len).value_counts().index,
            autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    plt.title('Proportion of Link Counts per Tweet')
    plt.savefig(f'{params.dal["statPath2Save"]}/proportion_link_counts_per_tweet.png')
    plt.close()




    # Return additional stats if needed
    return {
        'num_tweets': num_tweets,
        'num_users': num_users,
        'num_tweets_with_links': num_tweets_with_links,
        'unique_users_with_links': unique_users_with_links,
        'unique_users_without_links': unique_users_without_links,
        'total_links': total_links,
        'avg_tweets_per_user': avg_tweets_per_user,
        'max_tweets_by_user': max_tweets_by_user,
        'min_tweets_by_user': min_tweets_by_user,
        'user_counts': user_counts,
        'links_per_user': links_per_user
    }


