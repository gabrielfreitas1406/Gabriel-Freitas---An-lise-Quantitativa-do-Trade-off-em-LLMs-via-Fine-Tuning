import json
from datasets import load_dataset
import random

def preprocess_spider(save_path="processed_data"):
    # Carregar dataset
    spider = load_dataset("spider")
    
    # 1. Selecionar exemplos few-shot (3 exemplos fixos para consistência)
    few_shot_examples = spider["train"].select([10, 25, 42])  # Índices fixos para reprodutibilidade
    
    # 2. Processar dados de treino
    def process_train_example(example):
        # Criar prompt few-shot + exemplo atual
        prompt = create_prompt(example["question"], few_shot_examples, include_answer=True)
        return {"text": prompt}
    
    train_data = spider["train"].map(process_train_example)
    
    # 3. Processar dados de validação (sem incluir resposta no prompt)
    def process_val_example(example):
        prompt = create_prompt(example["question"], few_shot_examples, include_answer=False)
        return {
            "text": prompt,
            "db_id": example["db_id"],
            "query": example["query"],
            "question": example["question"]
        }
    
    val_data = spider["validation"].map(process_val_example)
    
    # 4. Salvar dados processados
    train_data.save_to_disk(f"{save_path}/train")
    val_data.save_to_disk(f"{save_path}/validation")
    
    # Salvar exemplos few-shot separadamente para uso posterior
    with open(f"{save_path}/few_shot_examples.json", "w") as f:
        json.dump(few_shot_examples.to_dict(), f)
    
    print(f"Dados processados salvos em {save_path}")

def create_prompt(question, few_shot_examples, include_answer=True):
    """
    Cria prompt no formato do modelo google/gemma-2b-it.
    """
    system_msg = "You are a SQL expert. Convert the following natural language questions to SQL queries."

    # Começa com uma instrução de sistema (opcional, mas ajuda a alinhar)
    prompt = (
        "<start_of_turn>user\n"
        f"{system_msg}\n"
        "<end_of_turn>\n"
    )

    # Adiciona exemplos few-shot
    for ex in few_shot_examples:
        prompt += (
            "<start_of_turn>user\n"
            f"Question: {ex['question']}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            f"SQL: {ex['query']}\n"
            "<end_of_turn>\n"
        )

    # Adiciona a pergunta atual
    prompt += (
        "<start_of_turn>user\n"
        f"Question: {question}\n"
        "<end_of_turn>\n"
    )

    if include_answer:
        prompt += "<start_of_turn>model\nSQL:"

    return prompt


if __name__ == "__main__":
    preprocess_spider()