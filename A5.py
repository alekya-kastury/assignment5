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

#Data preparation 
embed_model = "text-embedding-ada-002"

# connect to index
index = pinecone.Index(index_name)
# view index stats
#index.describe_index_stats()

st.title("Welcome")

query = st.text_input("How can I assist you today?",value='Type something here')


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

query_with_contexts = retrieve(query)

res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=query_with_contexts,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

st.write(res)

