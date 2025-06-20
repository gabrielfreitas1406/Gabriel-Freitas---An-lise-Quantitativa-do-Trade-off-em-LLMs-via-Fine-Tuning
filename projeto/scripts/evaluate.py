import sqlite3
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluator import evaluate
import torch

class SQLExecutionAccuracy(BaseMetric):
    def __init__(self, database_dir: str):
        super().__init__()
        self.database_dir = database_dir
        self.threshold = 0.5

    def measure(self, test_case: LLMTestCase) -> float:
        db_path = Path(self.database_dir) / f"{test_case.input_params['db_id']}.sqlite"
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Executar SQL gerado
            try:
                cursor.execute(test_case.actual_output)
                generated_result = set(cursor.fetchall())
            except Exception as e:
                return 0.0
            
            # Executar SQL esperado
            try:
                cursor.execute(test_case.expected_output)
                expected_result = set(cursor.fetchall())
            except Exception as e:
                return 0.0
            
            return 1.0 if generated_result == expected_result else 0.0
            
        finally:
            conn.close()

    @property
    def __name__(self):
        return "SQLExecutionAccuracy"

def evaluate_model(model_path: str, data_path: str, db_path: str):
    # Carregar modelo e tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Carregar dados de validação
    eval_data = load_from_disk(f"{data_path}/validation")
    
    # Preparar casos de teste
    test_cases = []
    for item in eval_data.select(range(50)):  # Avaliar apenas 50 exemplos
        input_text = item["text"]  # Já contém o prompt few-shot
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=1024)
        
        generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_sql = generated_sql.split("SQL:")[-1].strip()
        
        test_case = LLMTestCase(
            input=input_text,
            actual_output=generated_sql,
            expected_output=item["query"],
            context={"db_id": item["db_id"]}
        )
        test_cases.append(test_case)
    
    # Avaliar
    metric = SQLExecutionAccuracy(database_dir=db_path)
    evaluate(test_cases, [metric], print_results=True)

if __name__ == "__main__":
    evaluate_model(
        model_path="sql_llama_finetuned",
        data_path="processed_data",
        db_path="spider/database"  # Assumindo que os DBs do Spider estão aqui
    )