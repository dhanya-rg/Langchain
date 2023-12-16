import sys
import os
from constants import openai_key

from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI


# openai LLMs

os.environ["OPENAI_API_KEY"]=openai_key

query = sys.argv[1]
loader = TextLoader("data\data.txt")
print(query)

index=VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query,llm=ChatOpenAI()))