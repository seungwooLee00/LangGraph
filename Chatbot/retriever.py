from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents.base import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.base import Runnable
from typing import List
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def createRetriever(model, docs: List[Document]) -> Runnable:
  logging.info("Initializing Chroma vector store...")
  vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
  )
  
  child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
  parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
  pd_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
  )
  
  logging.info(f"Adding {len(docs)} documents to the retriever...")
  pd_retriever.add_documents(docs)
  
  llm_chain = _getLLMchain(model)
  mq_retriever = MultiQueryRetriever(
    retriever=pd_retriever,
    llm_chain=llm_chain,
    parser_key="lines"
  )
  
  logging.info("Retriever successfully created.")
  return mq_retriever

def _getLLMchain(model: ChatOpenAI) -> Runnable:
  output_parser = LineListOutputParser()
  QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate three 
      different versions of the given user question to retrieve relevant documents from a vector 
      database. Generate these alternative questions in both English and Korean to help the user 
      overcome limitations of distance-based similarity search due to mixed-language data.
      Provide these alternative questions separated by newlines.

      Original question (English): {question}
      """,
  )
  llm_chain = QUERY_PROMPT | model | output_parser
  
  return llm_chain
  
class LineListOutputParser(BaseOutputParser[List[str]]):
  """Output parser for a list of lines."""
  def parse(self, text: str) -> List[str]:
    lines = text.strip().split("\n")
    return list(filter(None, lines)) 