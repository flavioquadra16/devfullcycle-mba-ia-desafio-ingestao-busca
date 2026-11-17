from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit"}


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta:")
    print("Digite 'sair' para encerrar.\n")

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break

        try:
            answer = chain.invoke(question)
        except Exception as exc:
            print(f"Erro ao consultar: {exc}\n")
            continue

        print(f"RESPOSTA: {str(answer).strip()}\n")


if __name__ == "__main__":
    main()
