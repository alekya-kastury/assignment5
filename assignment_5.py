# -*- coding: utf-8 -*-
"""Assignment 5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/148iFu4QLkofkOxI5jdB0DyLOaw36QqWp

Assignment 5
"""
import streamlit as st
#!pip install git+https://github.com/openai/whisper.git -q

#!pip install ffmpeg-python

#!pip install -qU openai pinecone-client datasets tqdm

#!pip install pytube -q

import urllib.request

import pytube

import whisper

from pytube import YouTube

st.title("Welcome")

query = st.text_input("How can I assist you today?")

youtube_video_url='https://www.youtube.com/watch?v=tmGDx9hVWwo'

youtube=YouTube(youtube_video_url)

duration=youtube.length

yt_title = youtube.title

yt_channel_id=youtube.channel_id

yt_video_id=youtube.video_id

youtube.from_id

streams=youtube.streams.filter(only_audio=True)
stream=streams.first()
stream

yt_published_date=youtube.publish_date

stream.download(filename='output.mp4')

#!ffmpeg -ss 1 -i output.mp4 -t 3000 output_trimmed.mp4

model=whisper.load_model('base')

out=model.transcribe('output_trimmed.mp4')

out['segments']



data=[]
for segment in out['segments']:
  case1={'title':yt_title,
  'published':yt_published_date,
  'url':youtube_video_url,
  'video_id':yt_video_id,
  'channel_id':yt_channel_id,
  'id':str(yt_video_id)+'-t'+str(segment['start']),
  'text':segment['text'],
  'start':segment['start'],
  'end':segment['end']}
  data.append(case1)

text=[]
for a in range(3):
  text.append(data[a]['text'])

print (''.join(text))

text = ''.join([data[a]['text'] for a in range(3,7)])
print(text)

from tqdm.auto import tqdm

new_data = []

window = 20  # number of sentences to combine
stride = 4  # number of sentences to 'stride' over, used to create overlap

for i in tqdm(range(0, len(data), stride)):
    i_end = min(len(data)-1, i+window)
    #print(i)
    #print(i_end)
    if data[i]['title'] != data[i_end]['title']:
        # in this case we skip this entry as we have start/end of two videos
        continue
    text=''.join([data[a]['text'] for a in range(i,i_end)])
    new_data.append({
        'start': data[i]['start'],
        'end': data[i_end]['end'],
        'title': data[i]['title'],
        'text': text,
        'id': data[i]['id'],
        'url': data[i]['url'],
        'published': str(data[i]['published']),
        'channel_id': data[i]['channel_id']
    })

str(data[0]['published'])

new_data[0]

!pip install youtube_transcript_api 
!pip install openai
!pip install pinecone-client

from youtube_transcript_api import YouTubeTranscriptApi
import openai
import pinecone

# set up OpenAI API
openai.api_key = "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"
embed_model = "text-embedding-ada-002"

# set up Pinecone
index_name = "openai-youtube-transcriptions"
pinecone.init(api_key="245cbb4a-88ac-4794-a455-a39588737f92", environment="us-east1-gcp")
pinecone.whoami()

openai.Engine.list()  # check we have authenticated

index_name = 'openai-youtube-transcriptions'



query = "who was the 12th person on the moon and when did they land?"

# now query text-davinci-003 WITHOUT context
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

res['choices'][0]['text'].strip()

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

query = (
    "Which training method should I use for sentence transformers when " +
    "I only have pairs of related sentences?"
)

#complete(query)

embed_model = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine',
        metadata_config={'indexed': ['channel_id', 'published']}
    )
# connect to index
index = pinecone.Index(index_name)
# view index stats
index.describe_index_stats()



from tqdm.auto import tqdm
from time import sleep

batch_size = 100  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(new_data), batch_size)):
    # find end of batch
    i_end = min(len(new_data), i+batch_size)
    meta_batch = new_data[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    # cleanup metadata
    meta_batch = [{
        'start': x['start'],
        'end': x['end'],
        'title': x['title'],
        'text': x['text'],
        'url': x['url'],
        'published': x['published'],
        'channel_id': x['channel_id']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)

res = openai.Embedding.create(
    input=[query],
    engine=embed_model
)

# retrieve from Pinecone
xq = res['data'][0]['embedding']

# get relevant contexts (including the questions)
res = index.query(xq, top_k=2, include_metadata=True)

res



limit = 3750

def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt



# first we retrieve relevant items from Pinecone
query_with_contexts = retrieve(query)
query_with_contexts



# then we complete the context-infused query
complete(query_with_contexts)
