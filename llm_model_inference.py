import os
import multiprocessing
import torch
from dotenv import load_dotenv

from langchain.llms import LlamaCpp
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chains import LLMChain

from pdf_parser import pdf_to_ocr

load_dotenv()

# GPU Inference
cuda_available = 0
if (torch.cuda.is_available()):
    print ('Nvidia GPU detected!')
    os.environ['LLAMA_CPP_LIB'] = os.getenv('LLAMA_CPP_LIB', 'usr/local/lib/libllama.so')
    os.environ['LLAMA_CUBLAS'] = os.getenv('LLAMA_CUBLAS', 'on')
    cuda_available = 1
else:
    print ('Defaulting to CPU!')

# Model initialization
MODEL_PATH = os.getenv('MODEL_PATH')
if cuda_available:
    # GPU Layers = 25 acceptable for 4GB VRAM
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=25, max_tokens=2048, temperature=0)
else:
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=2048, n_threads=multiprocessing.cpu_count(), temperature=0)

SUMMARY_PROMPT_TEMPLATE = """
### System: 
You are an AI assistant. You will be given a task. You must generate a detailed and long answer.

### User:
Summarize the following text.
{text}

### Response:
Sure, here is a summary of the text:
"""

GRAMMAR_PROMPT_TEMPLATE = """
### System: 
You are an AI assistant that follows instruction extremely well. Help as much as you can.

### User:
Read the following text, and rewrite all sentences after correcting all the writing mistakes:
{text}

### Response:
"""

summarize_prompt = PromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
grammar_prompt = PromptTemplate(template=GRAMMAR_PROMPT_TEMPLATE, input_variables=["text"])

def summarize_pdf(pdf_path):
    # Convert pdf to text
    text = pdf_to_ocr(pdf_path)
    # Split text into chunks
    text_splitter = TokenTextSplitter()
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    summary_chain = load_summarize_chain(llm, chain_type='stuff', prompt=summarize_prompt)
    # Run inference
    try:
        result = summary_chain.run(docs)
    except Exception as e:
        return e

    return result

def grammar_check(text):
    llm_chain = LLMChain(prompt=grammar_prompt, llm=llm)
    # Run inference
    try:
        result = llm_chain.run(text)
        return result
    
    except Exception as e:
        return e


# Sample inference
if __name__ == "__main__":
    pdf_path = "./samples/tiny-attention.pdf"
    res = summarize_pdf(pdf_path)
    print (res)