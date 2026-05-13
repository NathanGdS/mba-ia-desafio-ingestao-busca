from search import search_prompt

def main():
    question = input("Faça sua pergunta:")
    chain = search_prompt(question)

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    
    pass

if __name__ == "__main__":
    main()