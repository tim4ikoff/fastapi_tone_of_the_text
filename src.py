import pandas as pd
import requests
import urllib.parse as urlparse
from tqdm import tqdm

import config


def get_video_id(url_value):
    query = urlparse.urlparse(url_value)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ['www.youtube.com', 'youtube.com']:
        if query.path == '/watch':
            p = urlparse.parse_qs(query.query)
            return p['v'][0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    return None


def get_comments(api_key, video_id):

    endpoint = 'https://www.googleapis.com/youtube/v3/commentThreads'
    params = {'part': 'snippet', 'videoId': video_id, 'maxResults': 100, 'key': api_key}
    response = requests.get(endpoint, params=params)
    res = response.json()

    if 'items' in res.keys():
        return {
            num: {
                'text_comment': ' '.join(
                    x['snippet']['topLevelComment']['snippet']['textOriginal'].splitlines()
                ),
                'publish_data': x['snippet']['topLevelComment']['snippet']['publishedAt'],
            }
            for num, x in enumerate(res['items'])
        }
    return None


def get_sentim(data, model):
    res = model(data)[0]
    return res['label'], res['score']


def pipeline_sentiment(url_value, api_key, model):

    video_id = get_video_id(url_value)
    result = get_comments(api_key, video_id)
    result = pd.DataFrame(result).T

    text_tuple = [get_sentim(i, model) for i in tqdm(result[config.col_text_comment])]
    result[[config.col_sentiment, 'score']] = pd.DataFrame(list(text_tuple))
    return result


def pipeline_stats(data, col_sentiment):
    return (data.groupby([col_sentiment])[[col_sentiment]].count() / data.shape[0])[
        config.col_sentiment
    ]


def pipeline_summarize(data, model, length=2057, max_length=70):
    text = '.'.join(data)
    result_text = []

    for i in tqdm(range(0, len(text), length)):
        new_text = text[i:i+length]
        result_text.append(model(new_text, max_length=max_length))

    print(result_text)
    return '. '.join([i[0]['summary_text'] for i in result_text])