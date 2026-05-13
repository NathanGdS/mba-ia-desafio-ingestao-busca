# Desafio MBA Engenharia de Software com IA - Full Cycle

Pipeline RAG: ingere PDF, armazena embeddings em PGVector e responde perguntas via LLM.

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API OpenAI ou Google Gemini

## Configuração

**1. Variáveis de ambiente**

```bash
cp .env.example .env
```

Edite `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small   # padrão

DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=documents

PDF_PATH=document.pdf   # caminho relativo à raiz do projeto
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

**2. Dependências Python**

```bash
python -m venv venv
##
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
##
pip install -r requirements.txt
```

## Execução

### 1. Subir o banco de dados

```bash
docker compose up -d
```

Aguarda o PostgreSQL ficar saudável e instala a extensão `vector` automaticamente.

### 2. Ingerir o PDF

```bash
python src/ingest.py
```

Carrega o PDF definido em `PDF_PATH`, divide em chunks e armazena os embeddings no PGVector.

### 3. Rodar o chat

```bash
python src/chat.py
```

Solicita uma pergunta, busca os chunks mais relevantes e retorna a resposta gerada pelo LLM.

---

### Atalhos via Make

```bash
make ingest   # equivale ao passo 2
make chat     # equivale ao passo 3
```
