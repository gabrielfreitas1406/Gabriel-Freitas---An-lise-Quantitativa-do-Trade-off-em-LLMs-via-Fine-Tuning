from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
import os
from datetime import datetime
from huggingface_hub import login

login(token=os.getenv("HUGGINGFACE_TOKEN"))

def load_processed_data(data_path="processed_data"):
    """Carrega os dados pré-processados do disco"""
    train_data = load_from_disk(f"{data_path}/train")
    val_data = load_from_disk(f"{data_path}/validation")
    return train_data, val_data

def print_trainable_parameters(model):
    """Imprime o número de parâmetros treináveis"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Parâmetros treináveis: {trainable_params} || Todos os parâmetros: {all_param} || "
        f"Percentual treinável: {100 * trainable_params / all_param:.2f}%"
    )

def train_model():

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Acelera download
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Reduz logs

    # 1. Configurações iniciais
    model_name =  "HuggingFaceH4/zephyr-7b-alpha" #"mistralai/Mistral-7B-Instruct-v0.2" #meta-llama/Llama-3-8B-Instruct
    output_dir = f"results/sql-zephyr-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Carregar dados processados
    train_data, val_data = load_processed_data()
    
    # 3. Configuração do modelo com quantização 4-bit
    device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0, 
        "model.layers.3": 0,
        "model.norm": "cpu",
        "lm_head": "cpu",
        "__": "cpu"
    }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True  
    )
    
    # Carregar modelo e tokenizador
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder="offload",
        offload_state_dict=True,
        #force_embed_tokens_to_device=0 if torch.cuda.is_available() else "cpu"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Preparar modelo para treinamento k-bit
    model = prepare_model_for_kbit_training(model)
    
    # 4. Configuração LoRA
    '''peft_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"]  # Camadas adicionais para adaptação
    )'''
    peft_config = LoraConfig(
        r=4,  
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],  # Apenas Q e V
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
        
    )
    
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    
    # 5. Configuração do treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        optim="paged_adamw_32bit",
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="tensorboard",
        save_total_limit=3,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4
    )
    
    # 6. Inicializar Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        neftune_noise_alpha=5  # Regularização NEFTune
    )
    
    # 7. Treinamento
    print("Iniciando treinamento...")
    trainer.train()
    
    # 8. Salvar modelo final
    final_model_dir = f"{output_dir}/final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print(f"Treinamento concluído! Modelo salvo em: {final_model_dir}")

if __name__ == "__main__":
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    train_model()