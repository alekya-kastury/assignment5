import streamlit as st
import urllib.request
import whisper
from pytube import YouTube

from tqdm.auto import tqdm
from time import sleep

from youtube_transcript_api import YouTubeTranscriptApi
import openai
import pinecone


st.title("Welcome")

query = st.text_input("How can I assist you today?")
youtube_video_url='https://www.youtube.com/watch?v=tmGDx9hVWwo'

youtube=YouTube(youtube_video_url)

duration=youtube.length

yt_title = youtube.title

yt_channel_id=youtube.channel_id

yt_video_id=youtube.video_id

streams=youtube.streams.filter(only_audio=True)
stream=streams.first()
stream

yt_published_date=youtube.publish_date
stream.download(filename='output.mp4')
model=whisper.load_model('base')

out=model.transcribe('output_trimmed.mp4')

#out['segments']
