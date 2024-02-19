import os
import multiprocessing
import torch
import chromadb
from dotenv import load_dotenv

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

from pdf_parser import pdf_to_ocr, pdf_to_ocr_raw

load_dotenv()

# GPU Inference
cuda_available = 0
if (torch.cuda.is_available()):
    print ('Nvidia GPU detected!')
    os.environ['LLAMA_CPP_LIB'] = os.getenv('LLAMA_CPP_LIB', 'usr/local/lib/libllama.so')
    os.environ['LLAMA_CUBLAS'] = os.getenv('LLAMA_CUBLAS', 'on')
    os.environ['GGML_CUDA_NO_PINNED'] = '1' # Pinned memory warning fix
    cuda_available = 1
else:
    print ('Defaulting to CPU!')

# Model initialization
MODEL_PATH = os.getenv('MODEL_PATH')
if cuda_available:
    # TODO: GPU Layers = 5 acceptable for 4GB VRAM?
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=32768, n_gpu_layers=15, n_batch=512, max_tokens=32768)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': False}
    )
else:
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=32768, n_threads=multiprocessing.cpu_count(), max_tokens=32768)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

# Chroma DB
chroma_client = chromadb.Client()

# Prompts
SUMMARY_PROMPT_TEMPLATE = """[INST]You are My-GPT, a helpful assistant who does summarization of research papers and news articles accurately. You need to summarize the text delimited by triple backticks, without adding any information of your own. Always keep your responses brief.
```{text}```[/INST]
"""

GRAMMAR_PROMPT_TEMPLATE = """[INST]You are My-GPT, a helpful assistant who does grammar checks and reformatting of text. Make MINIMUM edits to rectify the grammar in the text. You are NOT allowed to change what the text conveys through your reformatting. Only return the corrected text.
{text}[/INST]
"""

CHAT_PROMPT_TEMPLATE = """[INST]You are My-GPT, a helpful and friendly question-answering assistant who answers questions from the document given to you by the user. When any questions are asked to you from the document, respond accurately without adding any information of your own. When you don't find an answer from the document provided, respond "I'm sorry, but I was unable to find the answer". You are also given your previous conversation history with the user, where your responses are prefixed with the word "AI:" and the user's queries are prefixed with "Human:". Use it to continue the conversation and generate ONLY ONE response. Keep your responses crisp and clear.
Here is the document (sentences can start and end abruptly):
{context}

Conversation History:
{chat_history}

Human: {question}[/INST]
"""

summarize_prompt = PromptTemplate(template=SUMMARY_PROMPT_TEMPLATE, input_variables=["text"])
grammar_prompt = PromptTemplate(template=GRAMMAR_PROMPT_TEMPLATE, input_variables=["text"])
qa_prompt = PromptTemplate(template=CHAT_PROMPT_TEMPLATE, input_variables=["question", "chat_history", "context"])

def llm_chain_inference(prompt, text):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    result = llm_chain.run(text)
    return result

def summarize_pdf(pdf_path):
    try:
        text = pdf_to_ocr(pdf_path) # Convert pdf to text
        return llm_chain_inference(prompt=summarize_prompt, text=text) # Run inference
    except Exception as e:
        return e

def grammar_check(text):
    try:
        result = llm_chain_inference(prompt=grammar_prompt, text=text) # Run inference
        return result
    except Exception as e:
        return e

def ingest_file(pdf_path):
    # Convert pdf to text
    try:
        text = pdf_to_ocr_raw(pdf_path) # Prefered for chunking
        text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
        texts = text_splitter.split_text(text)

        docs = [Document(page_content=t) for t in texts]
        db = Chroma.from_documents(docs, embeddings, client=chroma_client)
        print ('File has been ingested!')
        return "File has been uploaded!"
    except Exception as e:
        return e

def chat_qa(query, chat_history):
    # Use stored embeddings
    db = Chroma(embedding_function=embeddings, client=chroma_client)
    print (db.get())

    # Initialize chat memory, uses chat state from frontend
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
    for index in range(0, len(chat_history), 2):
        prev_user_msg, prev_ai_msg = chat_history[index], chat_history[index+1]
        memory.chat_memory.add_user_message(prev_user_msg)
        memory.chat_memory.add_ai_message(prev_ai_msg)
    
    qa = load_qa_chain(llm=llm, memory=memory, prompt=qa_prompt, verbose=True) # Initialize QA
    docs = db.similarity_search(query, k=2) # Get relevant docs

    result = qa({
        "input_documents" : docs,
        "question" : query
    })
    print (result, flush=True)

    return result['output_text'].split('AI:')[-1].strip()

# Sample inference
if __name__ == "__main__":
    pdf_path = "./samples/tiny-attention.pdf"
    res = summarize_pdf(pdf_path)
    print (res)