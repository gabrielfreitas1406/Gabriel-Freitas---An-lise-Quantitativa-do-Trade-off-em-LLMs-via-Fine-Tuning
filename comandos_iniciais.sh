#!/bin/bash

# fix_dependencies_enhanced.sh - Script aprimorado para instalação de dependências LLM

# Configurações
LOG_FILE="dependency_install_enhanced.log"
PYTHON_CMD="python3"
PIP_CMD="pip3"
TORCH_CUDA="cu118"  # Altere para sua versão de CUDA (cu121, cu118, etc)

# Função para verificar espaço em disco
check_disk_space() {
    local required=$1
    local available=$(df -k --output=avail . | tail -1)
    
    if [ "$available" -lt "$((required * 1024))" ]; then
        echo "ERRO: Espaço em disco insuficiente. Necessário pelo menos ${required}GB."
        exit 1
    fi
}

# Inicializar arquivo de log
echo "Registro de instalação - $(date)" > $LOG_FILE
echo "--------------------------------" >> $LOG_FILE

# 1. Verificar espaço em disco (pelo menos 5GB recomendado)
check_disk_space 5

# 2. Remover pacotes problemáticos
echo "Removendo pacotes existentes..." | tee -a $LOG_FILE
$PIP_CMD uninstall -y triton bitsandbytes torch >> $LOG_FILE 2>&1
$PIP_CMD cache purge >> $LOG_FILE 2>&1

# 3. Instalar PyTorch com CUDA
echo "Instalando PyTorch com CUDA ${TORCH_CUDA}..." | tee -a $LOG_FILE
$PIP_CMD install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
    --no-cache-dir >> $LOG_FILE 2>&1 || {
    echo "Falha na instalação do PyTorch. Tentando instalação alternativa..." | tee -a $LOG_FILE
    $PIP_CMD install torch torchvision torchaudio >> $LOG_FILE 2>&1
}

# 4. Tentar instalar Triton de múltiplas fontes
install_triton() {
    echo "Tentando instalar Triton (método $1)..." | tee -a $LOG_FILE
    
    case $1 in
        1)
            $PIP_CMD install triton==2.0.0 --no-cache-dir >> $LOG_FILE 2>&1
            ;;
        2)
            $PIP_CMD install triton --pre >> $LOG_FILE 2>&1
            ;;
        3)
            $PIP_CMD install triton-nightly --upgrade >> $LOG_FILE 2>&1
            ;;
        *)
            return 1
            ;;
    esac
}

# Tentar diferentes métodos de instalação
for method in 1 2 3; do
    install_triton $method && break
done

# 5. Instalar bitsandbytes
echo "Instalando bitsandbytes..." | tee -a $LOG_FILE
$PIP_CMD install bitsandbytes --prefer-binary >> $LOG_FILE 2>&1 || {
    echo "Falha na instalação padrão. Tentando instalação alternativa..." | tee -a $LOG_FILE
    $PIP_CMD install https://github.com/jllllll/bitsandbytes/releases/download/0.41.1/bitsandbytes-0.41.1-py3-none-any.whl >> $LOG_FILE 2>&1
}

# 6. Instalar outras dependências
echo "Instalando transformers, datasets, peft, trl, accelerate..." | tee -a $LOG_FILE
$PIP_CMD install transformers datasets peft trl accelerate >> $LOG_FILE 2>&1

# Verificação final
echo -e "\nVerificando instalações:" | tee -a $LOG_FILE
$PYTHON_CMD -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import bitsandbytes as bnb; print(f'bitsandbytes: {bnb.__version__}, CUDA: {bnb.CUDA_SETUP["SUCCESS"] if hasattr(bnb, 'CUDA_SETUP') else 'N/A'}')
import triton; print(f'Triton: {triton.__version__}')
" 2>&1 | tee -a $LOG_FILE

echo -e "\nProcesso concluído. Verifique $LOG_FILE para detalhes." | tee -a $LOG_FILE