from huggingface_hub import notebook_login
import torch
import transformers
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                            #  torch_dtype=torch.float16,
                                            #  use_auth_token=True,
                                            #  load_in_8bit=True
                                            #  load_in_4bit=True
                                             )


pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                max_new_tokens = 512,
                eos_token_id=tokenizer.eos_token_id
                )

llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0.1})

prompt = PromptTemplate(
    input_variables = ['temp'],
    template = "{temp}, explain in steps."
)

chain = LLMChain(llm=llm, prompt=prompt)

chain.run("How to open a bank account")