#This is a demo code to show how we can use LLM chains for NLP data pre-processing in sequential manner and create a pipeline
#1 HTML tag removal
#2. Replace contractions in string
#3. Remove no.
#4. Tokenization
#5. Remove stopwords

import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from constants import openai_key
import gradio as gr
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = openai_key

def preprocess (text):
    
    llm = OpenAI(model='gpt-3.5-turbo-instruct',temperature=0.8)

    HTML_parsed_text_memory = ConversationBufferMemory(input_key='text',memory_key='HTML_parse_text_history_')
    Clean_text_memory = ConversationBufferMemory(input_key='HTML parsed text',memory_key='Clean_text_history_')
    Removed_Numbers_memory = ConversationBufferMemory(input_key='Clean text',memory_key='Remove.no_text_history_')
    Tokenized_text_memory = ConversationBufferMemory(input_key='Removed numbers',memory_key='Tokenized_text_history_')


    #Parse for HTML
    prompt_template_1 = PromptTemplate(
    input_variables=["text"],
    template="Instruct the model to remove HTML tags from the following text: {text}"
    )
    chain1 = LLMChain(llm=llm,prompt=prompt_template_1,verbose=True,output_key='HTML parsed text',memory=HTML_parsed_text_memory)

    #Replace contractions
    prompt_template_2 = PromptTemplate(
    input_variables=["HTML parsed text"],
    template="Instruct the model to replace contractions from the following text: {HTML parsed text}"
    )

    chain2 = LLMChain(llm=llm,prompt=prompt_template_2,verbose=True,output_key='Clean text',memory = Clean_text_memory)
    
    #Remove numbers
    prompt_template_3 = PromptTemplate(
    input_variables=["Clean text"],
    template="Instruct the model to remove any numbers from the following text: {Clean text}"
    )

    chain3 = LLMChain(llm=llm,prompt=prompt_template_3,verbose=True,output_key='Removed numbers',memory=Removed_Numbers_memory)
    
    #Tokenize the words
    prompt_template_4 = PromptTemplate(
    input_variables=["Removed numbers"],
    template="Instruct the model to tokenize the following text: {Removed numbers}"
    )

    chain4 = LLMChain(llm=llm,prompt=prompt_template_4,verbose=True,output_key='Tokenized text',memory = Tokenized_text_memory)
    
    #Remove stopwords and special characters
    prompt_template_5 = PromptTemplate(
    input_variables=["Tokenized text"],
    template="Instruct the model to remove any stopwords and special characters from the following text: {Tokenized text}"
    )
    
    chain5 = LLMChain(llm=llm,prompt=prompt_template_5,verbose=True,output_key='Final text')

    parent_chain = SequentialChain(chains = [chain1,chain2,chain3,chain4,chain5],input_variables=['text'],output_variables=['Final text'],verbose=True)
    response = parent_chain({'text':text})
    
    return response['Final text'].strip(),Clean_text_memory.buffer,Tokenized_text_memory.buffer

Response_output = gr.Textbox(label='Processed data')
Clean_output = gr.Textbox(label='Clean data')
Tokenized_output = gr.Textbox(label='Tokenized data')

interface = gr.Interface(fn=preprocess,
                         inputs=["textbox"],
                         title='Preprocessing application for NLP use cases',
                         outputs=[Response_output,Clean_output,Tokenized_output]) # Text output for the translated text

interface.launch()








