import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool

load_dotenv()
for k in ("OPENAI_API_KEY", "DATABASE_URL","PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.
- Nunca responda de forma "seca" e simplista, elabore em algumas frases a sua resposta.
- Nunca diga que você pegou sua resposta de algum outro material, responda como se você já soubesse por natureza a resposta.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

EXEMPLOS DE PERGUNTAS MAL FORMATADAS:

PERGUNTA: Qual o faturamento da Empresa xpto?
RESPOSTA MAL FORMATADA: Conforme o conteúdo fornecido, o faturamento da xpto é de R$ 10.000.000,00 para o ano de 2025.
RESPOSTA CORRETA: O faturamento da xpto foi de R$ 10.000.000,00 para o ano de 2025.

PERGUNTA: Qual o faturamento da Empresa xpto?
RESPOSTA MAL FORMATADA: Conforme o material fornecido, o faturamento da xpto é de R$ 10.000.000,00 para o ano de 2025.
RESPOSTA CORRETA: O faturamento da xpto foi de R$ 10.000.000,00 para o ano de 2025.

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(question=None):
    # tools = [search]
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small"))

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
    context = store.similarity_search_with_score(question, k=3)

    template = PromptTemplate(
        input_variables=["pergunta", "contexto"],
        template=PROMPT_TEMPLATE
    )
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.9)


    pipeline = template | llm | StrOutputParser()

    return pipeline.invoke({"pergunta": question, "contexto": context})