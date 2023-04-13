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
data = YouTubeTranscriptApi.get_transcript('tmGDx9hVWwo')
st.write(data)
