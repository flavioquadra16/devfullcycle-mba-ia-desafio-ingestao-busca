# Ingestão e Busca Semântica com LangChain e Postgres

Aplicação que lê um PDF, grava os trechos como embeddings no PostgreSQL com pgVector e responde perguntas no terminal usando somente o conteúdo ingerido.

## O que o projeto faz

1. **Ingestão** (`src/ingest.py`): carrega `document.pdf`, divide o texto em chunks de 1000 caracteres com overlap de 150, gera embeddings e salva os vetores no Postgres.
2. **Busca** (`src/search.py`): vetoriza a pergunta, busca os 10 trechos mais próximos (`similarity_search_with_score`, `k=10`) e monta o prompt exigido pelo desafio.
3. **Chat** (`src/chat.py`): loop no terminal. Se a resposta não estiver explícita no contexto recuperado, o modelo deve responder: `Não tenho informações necessárias para responder sua pergunta.`

## Modelos

O desafio não fixa modelos. Esta entrega usa, por padrão, os modelos leves da OpenAI que continuam disponíveis e cabem no volume do PDF:

| Uso | Variável | Padrão |
| --- | --- | --- |
| Embeddings | `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` |
| Resposta | `OPENAI_LLM_MODEL` | `gpt-4o-mini` |

Também é possível usar Gemini, com os nomes configuráveis:

| Uso | Variável | Padrão |
| --- | --- | --- |
| Embeddings | `GOOGLE_EMBEDDING_MODEL` | `models/gemini-embedding-001` |
| Resposta | `GOOGLE_LLM_MODEL` | `gemini-flash-lite-latest` |

Defina `LLM_PROVIDER=openai` ou `LLM_PROVIDER=gemini`. Ingestão e chat precisam usar o mesmo provedor e o mesmo modelo de embeddings. A tabela de vetores é criada na primeira ingestão com a dimensão desse modelo. Se o modelo de embeddings mudar depois, apague o volume do banco e rode a ingestão de novo:

```bash
docker compose down -v
docker compose up -d
python src/ingest.py
```

Consulte a documentação oficial antes de trocar os nomes, porque modelos são descontinuados com frequência:

- OpenAI: https://platform.openai.com/docs/models
- Gemini: https://ai.google.dev/gemini-api/docs/models

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Chave de API da OpenAI ou do Google AI Studio

## Configuração

```bash
python -m venv venv
```

Linux ou macOS:

```bash
source venv/bin/activate
```

Windows (PowerShell):

```powershell
venv\Scripts\Activate.ps1
```

Instale as dependências e crie o arquivo de ambiente:

```bash
pip install -r requirements.txt
copy .env.example .env
```

No Linux ou macOS, use `cp .env.example .env`. Edite o `.env` e preencha a chave do provedor escolhido. A URL do banco já aponta para o Postgres do `docker-compose.yml`:

```text
postgresql+psycopg://postgres:postgres@localhost:5432/rag
```

## Execução

Suba o banco:

```bash
docker compose up -d
```

Ingerir o PDF:

```bash
python src/ingest.py
```

Abrir o chat:

```bash
python src/chat.py
```

Exemplo:

```text
Faça sua pergunta:

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.
```

Perguntas que não estão no PDF:

```text
PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Digite `sair` para encerrar o chat. Rodar a ingestão de novo apaga a collection anterior e grava os chunks outra vez, para não duplicar vetores.

## Estrutura

```text
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── settings.py       # Provedor, conexão e factories
│   ├── ingest.py         # Ingestão do PDF
│   ├── search.py         # Busca vetorial e prompt
│   ├── chat.py           # CLI
├── document.pdf
└── README.md
```
