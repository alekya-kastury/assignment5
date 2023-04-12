import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import pinecone as p
#from pinecone import init

# set up OpenAI API
openai.api_key = st.secrets["openai_api_key"]
embed_model = "text-embedding-ada-002"

# set up Pinecone
index_name = "openai-youtube-transcriptions"
p.init(api_key="245cbb4a-88ac-4794-a455-a39588737f92", environment="us-east1-gcp")

# define function to get transcript data
def get_transcript_data(video_id):
    data = YouTubeTranscriptApi.get_transcript(video_id)
    new_data = []
    window = 20  # number of sentences to combine
    stride = 4  # number of sentences to 'stride' over, used to create overlap

    for i in range(0, len(data), stride):
        i_end = min(len(data) - 1, i + window)
        if data[i]["title"] != data[i_end]["title"]:
            # in this case we skip this entry as we have start/end of two videos
            continue
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


# set up Streamlit
st.title("YouTube Transcript Embeddings Search")

video_id = st.text_input("Enter a YouTube video ID:")
if video_id:
    # load transcript data
    transcript_data = get_transcript_data(video_id)

    # define search query
    query = st.text_input("Enter a search query:")

