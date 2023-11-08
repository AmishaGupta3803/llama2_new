from huggingface_hub import notebook_login
import torch
import transformers
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
from streamlit_chat import message


if 'tokenizer' not in st.session_state:
  st.session_state.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token='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')

if 'model' not in st.session_state:
  st.session_state.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                            #  torch_dtype=torch.float16,
                                            #  use_auth_token=True,
                                            #  load_in_8bit=True
                                            #  load_in_4bit=True,
                                             token='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'
                                             )


if 'pipe' not in st.session_state:
  st.session_state.pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                max_new_tokens = 512,
                eos_token_id=tokenizer.eos_token_id
                )

if 'llm' not in st.session_state:
  st.session_state.llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0.1})

prompt = PromptTemplate(
    input_variables = ['temp'],
    template = "{temp}, explain in steps."
)

chain = LLMChain(llm=llm, prompt=prompt)
user_input = st.chat_input()
if user_input:
    message(user_input, is_user=True)
    response = chain.run(user_input)
    message(response, is_user=False)
