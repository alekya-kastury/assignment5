import streamlit as st
import urllib.request
import whisper
from pytube import YouTube

from tqdm.auto import tqdm
from time import sleep

from youtube_transcript_api import YouTubeTranscriptApi
import openai
import pinecone

# set up OpenAI API
openai.api_key = st.secrets["openai_api_key"]
embed_model = "text-embedding-ada-002"

# set up Pinecone
index_name = "openai-youtube-transcriptions"
pinecone.init(api_key="245cbb4a-88ac-4794-a455-a39588737f92", environment="us-east1-gcp")

#youtube_video_url='https://www.youtube.com/watch?v=tmGDx9hVWwo'

data = YouTubeTranscriptApi.get_transcript('tmGDx9hVWwo')
data = YouTubeTranscriptApi.get_transcript('HeMIZC2rkMo')
data = YouTubeTranscriptApi.get_transcript('uFUGJnKByx8')
data = YouTubeTranscriptApi.get_transcript('D8ft1stIlCY')
data = YouTubeTranscriptApi.get_transcript('s0CIXvpfv5o')


def get_transcript_data(video_id):
    data = YouTubeTranscriptApi.get_transcript(video_id)
    new_data = []
    window = 20  # number of sentences to combine
    stride = 4  # number of sentences to 'stride' over, used to create overlap

    for i in range(0, len(data), stride):
        i_end = min(len(data) - 1, i + window)
        #if data[i]["title"] != data[i_end]["title"]:
            # in this case we skip this entry as we have start/end of two videos
         #   continue
        text = " ".join([d["text"] for d in data[i:i_end]])
        # create the new merged dataset
        new_data.append(
            {
                "start": data[i]["start"],
                "end": data[i_end]["end"],
                "title": data[i]["title"],
                "text": text,
                "id": data[i]["id"],
                "url": data[i]["url"],
                "channel_id": data[i]["channel_id"],
            }
        )

    # embed transcript data
    batch_size = 100  # how many embeddings we create and insert at once
    for i in range(0, len(new_data), batch_size):
        i_end = min(len(new_data), i + batch_size)
        meta_batch = new_data[i:i_end]
        # get ids
        ids_batch = [x["id"] for x in meta_batch]
        # get texts to encode
        texts = [x["text"] for x in meta_batch]
        # create embeddings
        res = openai.Embedding.create(input=texts, engine=embed_model)
        embeds = [record["embedding"] for record in res["data"]]
        # cleanup metadata
        meta_batch = [
            {
                "start": x["start"],
                "end": x["end"],
                "title": x["title"],
                "text": x["text"],
                "url": x["url"],
                "channel_id": x["channel_id"],
            }
            for x in meta_batch
        ]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        pinecone_index.upsert(vectors=to_upsert)

    return new_data

#video_id='tmGDx9hVWwo'
#transcript_data = get_transcript_data(video_id)

#def complete(prompt):
    # query text-davinci-003

st.title("Welcome")

query = st.text_input("How can I assist you today?")

res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=query,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

try:
    st.write(res['choices'][0]['text'])
except:
    st.write("Please enter a query")
