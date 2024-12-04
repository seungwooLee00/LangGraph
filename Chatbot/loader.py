from langchain_community.document_loaders.pdf import PDFPlumberLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents.base import Document
from tqdm import tqdm
from typing import List
import os
import logging
from django.conf import settings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

class Loader:
  def __init__(self):
    pass
  
  def invoke(self, data_path) -> List[Document]:
    csv_files = []
    md_files = []
    pdf_files = []

    for root, _, files in os.walk(data_path):
      for file in files:
        if file.endswith(".csv"):
          csv_files.append(os.path.join(root, file))
        elif file.endswith(".md"):
          md_files.append(os.path.join(root, file))
        elif file.endswith(".pdf"):
          pdf_files.append(os.path.join(root, file))
    
    logging.info(f"Found {len(csv_files)} csv files ")
    logging.info(f"Found {len(md_files)} markdown files ")
    logging.info(f"Found {len(pdf_files)} pdf files ")

    csv = Loader.readCSV(csv_files)
    md = Loader.readMarkdown(md_files)
    pdf = Loader.readPDF(pdf_files)
    docs = csv + md + pdf
    
    logging.info(f"create {len(docs)} docuements")
    return docs
  
  @staticmethod
  def readPDF(pdf_files) -> List[Document]:  
    docs = []
    for pdf_path in tqdm(pdf_files, desc="Reading PDFs", unit="file"):
      loader = PDFPlumberLoader(pdf_path)
      doc = loader.load()
      docs.extend(doc)

    return docs
  
  @staticmethod
  def readMarkdown(md_files) -> List[Document]:
    docs = []
    for md_path in tqdm(md_files, desc="Reading Markdowns", unit="file"):
      loader = UnstructuredMarkdownLoader(md_path)
      doc = loader.load()
      docs.extend(doc)

    return docs
  
  @staticmethod
  def readCSV(csv_files) -> List[Document]:
    docs = []
    for csv_path in tqdm(csv_files, desc="Reading CSV", unit="file"):
      loader = CSVLoader(csv_path)
      doc = loader.load()
      docs.extend(doc)
    
    return docs
      