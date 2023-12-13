"""url = https://www.youtube.com/watch?v=ScOf8Wh6m0w"""


from fastapi import FastAPI
import pandas as pd
import config
from src import pipeline_sentiment, pipeline_stats, pipeline_summarize
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import os

model_sent = pipeline(model=config.model_sent)
model_summ = pipeline(model=config.model_sum, use_fast=False)

app = FastAPI()


class YouTubeUrl(BaseModel):
    url_video: str


@app.post('/get_data')
def get_data(youtube: YouTubeUrl):
    data = pipeline_sentiment(youtube.url_video, config.API_KEY, model_sent)
    data.to_csv(f'{config.PATH_DATA}/{config.NAME_DATA}', index=False)
    return {'message': 'Successfully'}


@app.post('/get_stats_sent')
def get_stats_sent():
    if f'{config.NAME_DATA}' in os.listdir(f'{config.PATH_DATA}'):
        data = pd.read_csv(f'{config.PATH_DATA}/{config.NAME_DATA}')
        if config.col_sentiment == data.columns[2]:
            return pipeline_stats(data, config.col_sentiment)


@app.post('/get_summarize')
def get_summarize():
    if f'{config.NAME_DATA}' in os.listdir(f'{config.PATH_DATA}'):
        data = pd.read_csv(f'{config.PATH_DATA}/{config.NAME_DATA}')
        if config.col_text_comment == data.columns[0]:
            return pipeline_summarize(data[config.col_text_comment], model_summ)


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
