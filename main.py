## Integrate code with OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st
import gradio as gr
import openai

os.environ['OPENAI_API_KEY'] = openai_key


def chat_gpt(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

interface = gr.Interface(
  fn=chat_gpt,
  inputs="text",
  outputs="text",
  title="Query OpenAI LLM",
  description="Enter your search topic here!",
)

# Launch the interface
interface.launch()

#Gradio framework




