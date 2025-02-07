from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from langchain_google_vertexai import VectorSearchVectorStore
from langchain_google_vertexai import ChatVertexAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
 
 
# Configuração do projeto
 
PROJECT_ID = "folkloric-light-448714-u2"
REGION = "us-central1"
INDEX_ID = "2404143746783379456" # Substitua pelo ID do índice correto
INDEX_ENDPOINT_ID = "5394006133776056320" # Substitua pelo ID do endpoint correto
GCS_BUCKET_NAME = "processing-pdf-b" # Nome do bucket do GCS
 
 
# Inicializa o cliente do Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)
 
# Criando um modelo de embeddings do Vertex AI usando TextEmbeddingModel
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
 
 
# Função para gerar embeddings da consulta do usuário
def generate_query_embedding(query):
 embedding = embedding_model.get_embeddings([query])
 return embedding[0].values # Retorna o vetor de embeddings da consulta
 
# Conectando-se ao Vector Search do Matching Engine
 
vectorstore = VectorSearchVectorStore.from_components(
 project_id=PROJECT_ID,
 region=REGION,
 index_id=INDEX_ID,
 endpoint_id=INDEX_ENDPOINT_ID,
 gcs_bucket_name=GCS_BUCKET_NAME,
 embedding_model=generate_query_embedding
)
 
 
# Criando um retriever para buscar os documentos mais relevantes
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
 
print("Retriever criado com sucesso")
 
# Criando o modelo de linguagem do LangChain usando Vertex AI
llm = ChatVertexAI(model_name="gemini-pro", temperature=0.7)
 
print("LLM Carregada com sucesso")
 
# Criando um template de prompt para estruturar a RAG
prompt_template = PromptTemplate(
 
 template="""Você é um assistente inteligente. Use as informações abaixo para responder à pergunta:
 Contexto:
 {context}
 Pergunta do usuário:
 {question}
 Resposta:""",
 input_variables=["context", "question"]
)
 
# Criando a RAG Chain usando LangChain
qa_chain = RetrievalQA.from_chain_type(
 llm=llm,
 retriever=retriever,
 chain_type="stuff",
 chain_type_kwargs={"prompt": prompt_template}
)
 
# Exemplo de pergunta
query = "complete a frase: Declar a o CEDENTE que..."
response = qa_chain.invoke(query)
 
print("\nResposta da RAG:\n", response['result'])

import os
import json
import re
import uuid
 
from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
 
import PyPDF2
 
# Configurações do projeto
 
PROJECT_ID = os.getenv("PROJECT_ID", "folkloric-light-448714-u2")
REGION = os.getenv("REGION", "us-central1")
 
PROCESSING_BUCKET_NAME = os.getenv("PROCESSING_BUCKET_NAME")
 
 
# Obter variáveis de ambiente do evento do Cloud Storage
 
BUCKET_NAME = os.getenv("BUCKET_NAME")
PDF_FILE = os.getenv("PDF_FILE")
 
if not BUCKET_NAME or not PDF_FILE:
 raise ValueError("Variáveis de ambiente BUCKET_NAME e PDF_FILE são obrigatórias.")
 
# Gerar UID para identificação única do job
 
uid = str(uuid.uuid4())[:5]
 
embed_file_path = f"llmops_embedding_{uid}.json"
sentence_file_path = f"llmops_sentences_{uid}.json"
index_name = f"llmops_index_{uid}"
 
INDEX_ENDPOINT_NAME = "vector_search_endpoint" # Nome fixo do Index Endpoint
 
storage_client = storage.Client()
 
def download_pdf(bucket_name, pdf_file):
 
 """Faz o download do PDF do Cloud Storage."""
 
 bucket = storage_client.bucket(bucket_name)
 
 blob = bucket.blob(pdf_file)
 
 local_path = f"/tmp/{pdf_file}"
 
 blob.download_to_filename(local_path)
 
 print(f"Arquivo baixado: {local_path}")
 
 return local_path
 
def extract_sentences_from_pdf(pdf_path):
 
 """Extrai o texto do PDF e divide em sentenças."""
 
 with open(pdf_path, "rb") as file:
 
 reader = PyPDF2.PdfReader(file)
 
 text = " ".join([page.extract_text() or "" for page in reader.pages])
 
 return [sentence.strip() for sentence in text.split(". ") if sentence.strip()]
 
 
 
 
def generate_text_embeddings(sentences):
 
 """Gera embeddings para as sentenças usando Vertex AI."""
 
 model = TextEmbeddingModel.from_pretrained("text-embedding-004")
 
 embeddings = model.get_embeddings(sentences)
 
 return [embedding.values for embedding in embeddings]
 
def clean_text(text):
 
 """Limpa o texto removendo caracteres indesejados."""
 
 cleaned_text = re.sub(r"\u2022", "", text) # Remove bullet points
 
 return re.sub(r"\s+", " ", cleaned_text).strip() # Remove espaços extras
 
def save_embeddings(sentences, embeddings):
 
 """Salva os embeddings e sentenças em arquivos JSON."""
 
 with open(embed_file_path, "w") as embed_file, open(sentence_file_path, "w") as sentence_file:
 
 for sentence, embedding in zip(sentences, embeddings):
 
 json.dump({"id": uid, "embedding": embedding}, embed_file)
 
 embed_file.write("\n")
 
 json.dump({"id": uid, "sentence": clean_text(sentence)}, sentence_file)
 
 sentence_file.write("\n")
 
 print(f"Arquivos {embed_file_path} e {sentence_file_path} salvos.")
 
def upload_file(bucket_name, file_path):
 
 """Faz upload de um arquivo para o Cloud Storage."""
 
 bucket = storage_client.bucket(bucket_name)
 
 blob = bucket.blob(os.path.basename(file_path))
 
 blob.upload_from_filename(file_path)
 
 print(f"Arquivo {file_path} enviado para gs://{bucket_name}/{file_path}")
 
def create_vector_index(index_name):
 
 """Cria um índice novo e adiciona ao mesmo Index Endpoint."""
 
 aiplatform.init(project=PROJECT_ID, location=REGION)
 
 # Criar um novo índice para cada execução
 
 my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
 
 display_name=index_name,
 
 contents_delta_uri=f"gs://{PROCESSING_BUCKET_NAME}/{embed_file_path}",
 
 dimensions=768,
 
 approximate_neighbors_count=10,
 
 project=PROJECT_ID,
 
 location=REGION,
 )
 
 print(f"Índice {index_name} criado com sucesso.")
 
 # Verificar se o endpoint já existe
 
 endpoints = aiplatform.MatchingEngineIndexEndpoint.list(project=PROJECT_ID, location=REGION)
 
 existing_endpoint = None
 
 for endpoint in endpoints:
 
 if endpoint.display_name == INDEX_ENDPOINT_NAME:
 
 existing_endpoint = endpoint
 
 break
 
 # Criar o endpoint apenas se não existir
 
 if not existing_endpoint:
 
 existing_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
 
 display_name=INDEX_ENDPOINT_NAME,
 
 public_endpoint_enabled=True,
 
 project=PROJECT_ID,
 
 location=REGION,
 
 )
 
 print(f"Novo endpoint {INDEX_ENDPOINT_NAME} criado.")
 
 
 # Deployar o novo índice no endpoint existente
 
 deployed_index_id = f"deployed_{uid}"
 
 existing_endpoint.deploy_index(
 
 index=my_index,
 
 deployed_index_id=deployed_index_id,
 
 display_name=index_name,
 
 machine_type="n1-standard-16",
 
 min_replica_count=1,
 
 max_replica_count=1,
 
 )
 
 print(f"Índice {index_name} implantado com sucesso no endpoint {INDEX_ENDPOINT_NAME}.")
 
 
# Executa o processo
 
pdf_local_path = download_pdf(BUCKET_NAME, PDF_FILE)
sentences = extract_sentences_from_pdf(pdf_local_path)
embeddings = generate_text_embeddings(sentences)
save_embeddings(sentences, embeddings)
 
 
upload_file(PROCESSING_BUCKET_NAME, embed_file_path)
upload_file(PROCESSING_BUCKET_NAME, sentence_file_path)
 
create_vector_index(index_name)
 
print("Job concluído com sucesso.")


from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from langchain_google_vertexai import VectorSearchVectorStore
from langchain_google_vertexai import ChatVertexAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
 
 
# Configuração do projeto
 
PROJECT_ID = "folkloric-light-448714-u2"
REGION = "us-central1"
INDEX_ID = "2404143746783379456" # Substitua pelo ID do índice correto
INDEX_ENDPOINT_ID = "5394006133776056320" # Substitua pelo ID do endpoint correto
GCS_BUCKET_NAME = "processing-pdf-b" # Nome do bucket do GCS
 
 
# Inicializa o cliente do Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)
 
# Criando um modelo de embeddings do Vertex AI usando TextEmbeddingModel
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
 
 
# Função para gerar embeddings da consulta do usuário
def generate_query_embedding(query):
 embedding = embedding_model.get_embeddings([query])
 return embedding[0].values # Retorna o vetor de embeddings da consulta
 
# Conectando-se ao Vector Search do Matching Engine
 
vectorstore = VectorSearchVectorStore.from_components(
 project_id=PROJECT_ID,
 region=REGION,
 index_id=INDEX_ID,
 endpoint_id=INDEX_ENDPOINT_ID,
 gcs_bucket_name=GCS_BUCKET_NAME,
 embedding_model=generate_query_embedding
)
 
 
# Criando um retriever para buscar os documentos mais relevantes
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
 
print("Retriever criado com sucesso")
 
# Criando o modelo de linguagem do LangChain usando Vertex AI
llm = ChatVertexAI(model_name="gemini-pro", temperature=0.7)
 
print("LLM Carregada com sucesso")
 
# Criando um template de prompt para estruturar a RAG
prompt_template = PromptTemplate(
 
 template="""Você é um assistente inteligente. Use as informações abaixo para responder à pergunta:
 Contexto:
 {context}
 Pergunta do usuário:
 {question}
 Resposta:""",
 input_variables=["context", "question"]
)
 
# Criando a RAG Chain usando LangChain
qa_chain = RetrievalQA.from_chain_type(
 llm=llm,
 retriever=retriever,
 chain_type="stuff",
 chain_type_kwargs={"prompt": prompt_template}
)
 
# Exemplo de pergunta
query = "complete a frase: Declar a o CEDENTE que..."
response = qa_chain.invoke(query)
 
print("\nResposta da RAG:\n", response['result'])