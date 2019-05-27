import re
import nltk
from nltk.corpus import stopwords
import string

EMOTICONS_HAPPY = set([
':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
'=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
'<3'
])
EMOTICONS_SAD = set([
':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
':c', ':{', '>:\\', ';('
])
EMOJI_PATTERN = re.compile("["
     u"\U0001F600-\U0001F64F"  # emoticons
     u"\U0001F300-\U0001F5FF"  # symbols & pictographs
     u"\U0001F680-\U0001F6FF"  # transport & map symbols
     u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
     u"\U00002702-\U000027B0"
     u"\U000024C2-\U0001F251"
     "]+", flags=re.UNICODE)

EMOTICONS = EMOTICONS_HAPPY.union(EMOTICONS_SAD)
MORE_STOPWORDS = set(['http', 'https', 'co', 'vía', 'rt',
                      'así', 'por', 'tan', 'si', 'según',
                      'tras',
                      'via', '``', '`'])
tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

def get_tweets(api=None, screen_name=None):
    timeline = api.GetUserTimeline(screen_name=screen_name, count=200)
    earliest_tweet = min(timeline, key=lambda x: x.id).id

    while True:
        tweets = api.GetUserTimeline(
            screen_name=screen_name, max_id=earliest_tweet, count=200
        )
        new_earliest = min(tweets, key=lambda x: x.id).id
        if not tweets or new_earliest == earliest_tweet:
            break
        else:
            earliest_tweet = new_earliest
            #            print("getting tweets before:", earliest_tweet)
            timeline += tweets
    return timeline

def clean_tweet(tweet):
    stop_words = set(stopwords.words('english')).union(stopwords.words('spanish')).union(MORE_STOPWORDS)
    word_tokens = nltk.word_tokenize(tweet)

    #after tweepy preprocessing the colon symbol left remain after
    #removing mentions
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    #replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)

    #remove emojis from tweet
    tweet = EMOJI_PATTERN.sub(r'', tweet)

    #filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []

    #looping through conditions
    for w in word_tokens:
        #check tokens against stop words , emoticons and punctuations
        if w.lower() not in stop_words and w not in EMOTICONS and w not in string.punctuation and not w.startswith('//t.co'):
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)