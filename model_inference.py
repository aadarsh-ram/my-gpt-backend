import os
import subprocess

from langchain.llms import LlamaCpp
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chains.llm import LLMChain

from pdf_parser import pdf_to_ocr

# GPU Inference
cuda_available = 0
try:
    subprocess.run("nvidia-smi")
    print ('Nvidia GPU detected!')
    os.environ['LLAMA_CPP_LIB'] = os.getenv('LLAMA_CPP_LIB', 'usr/local/lib/libllama.so')
    os.environ['LLAMA_CUBLAS'] = os.getenv('LLAMA_CUBLAS', 'on')
    cuda_available = 1
except:
    print ('Defaulting to CPU!')

# Model initialization
MODEL_PATH = "./orca-mini-3b.q4_0.gguf"
if cuda_available:
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=25, max_tokens=2048, temperature=0.7)
else:
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=2048, max_tokens=2048, n_threads=8, temperature=0.7)

PROMPT_TEMPLATE = """
### System: 
You are an AI assistant that follows instruction extremely well. Help as much as you can.

### User:
Summarize the following text in your own words. Highlight the main contributions of the paper.
{text}

### Response:
Sure, here it is:
"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

def summarize_pdf(pdf_path):
    # Convert pdf to text
    text = pdf_to_ocr(pdf_path)
    # Split text into chunks
    text_splitter = TokenTextSplitter()
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    summary_chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt)
    # Run inference
    try:
        result = summary_chain.run(docs)
    except Exception as e:
        return e

    return result

# Sample inference
if __name__ == "__main__":
    pdf_path = "./samples/tiny-attention.pdf"
    res = summarize_pdf(pdf_path)
    print (res)