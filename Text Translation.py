import os
import openai
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from constants import openai_key

import gradio as gr

os.environ['OPENAI_API_KEY'] = openai_key

template = """Instruct the model to translate the following text:
{text} to {language} while maintaining the original meaning and tone"""

prompt_template = PromptTemplate(
  input_variables = ["text","language"],
  template = template
  
)

def translate (text,language):
    prompt=prompt_template.format(text=text,language = language)
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60
        )
    return response.choices[0].message.content.strip()


interface = gr.Interface(fn=translate,
                         inputs=["textbox", 
                                 gr.Dropdown(choices=["English", "Italian", "French", "German", "Spanish"]) 
    ],
    outputs="text"  # Text output for the translated text
)

interface.launch()