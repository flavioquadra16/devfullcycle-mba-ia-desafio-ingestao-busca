from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from settings import build_llm, build_vectorstore

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _join_results(results) -> str:
    parts = []
    for document, _score in results:
        text = (document.page_content or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def search_prompt(question=None):
    try:
        vectorstore = build_vectorstore()
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        def retrieve(query: str) -> str:
            results = vectorstore.similarity_search_with_score(query, k=10)
            return _join_results(results)

        chain = (
            {
                "contexto": RunnableLambda(retrieve),
                "pergunta": RunnablePassthrough(),
            }
            | prompt
            | build_llm()
            | StrOutputParser()
        )
    except Exception as exc:
        print(f"Erro de inicialização: {exc}")
        return None

    if question:
        return chain.invoke(question)
    return chain
