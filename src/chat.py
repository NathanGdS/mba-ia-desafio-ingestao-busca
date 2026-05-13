from search import search_prompt

def main():
    question = input("Faça sua pergunta: ")
    response = search_prompt(question)

    if not response:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    
    print(f"RESPOSTA: {response}")
    pass

if __name__ == "__main__":
    main()