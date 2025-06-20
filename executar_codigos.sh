#!/bin/bash

# run_pipeline.sh - Script para executar o pipeline de fine-tuning do modelo Spider

# Configurações
PROJECT_DIR="./projeto/scripts"
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# Função para verificar comandos
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Erro: $1 não está instalado. Por favor, instale antes de continuar."
        exit 1
    fi
}

# Função para criar diretórios se não existirem
create_dirs() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${PROJECT_DIR}/processed_data"
}

# Função para verificar ambiente Python
check_python() {
    if ! python -c "import torch, transformers, datasets, peft, trl" &> /dev/null; then
        echo "Erro: Dependências Python não encontradas. Ative o ambiente virtual primeiro."
        exit 1
    fi
}

# Início do script
echo "Iniciando pipeline de fine-tuning em: $(date)" | tee -a "${LOG_FILE}"

# Verificar comandos essenciais
check_command python3
check_command pip

# Criar estrutura de diretórios
create_dirs

# Verificar dependências Python
echo "Verificando dependências Python..." | tee -a "${LOG_FILE}"
check_python

# Executar pré-processamento
echo "Executando pré-processamento dos dados..." | tee -a "${LOG_FILE}"
cd "${PROJECT_DIR}" || { echo "Falha ao acessar ${PROJECT_DIR}"; exit 1; }

if ! python preprocess_spider.py >> "../${LOG_FILE}" 2>&1; then
    echo "Erro durante o pré-processamento. Verifique ${LOG_FILE} para detalhes." | tee -a "../${LOG_FILE}"
    exit 1
fi

# Executar treinamento
echo "Iniciando o processo de fine-tuning..." | tee -a "../${LOG_FILE}"

if ! python train.py >> "../${LOG_FILE}" 2>&1; then
    echo "Erro durante o treinamento. Verifique ${LOG_FILE} para detalhes." | tee -a "../${LOG_FILE}"
    exit 1
fi

# Finalização
echo "Pipeline concluído com sucesso em: $(date)" | tee -a "../${LOG_FILE}"
echo "Logs detalhados disponíveis em: ${LOG_FILE}" | tee -a "../${LOG_FILE}"