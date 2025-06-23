#!/bin/bash

# Script para instalar dependências do projeto SQL LLM Fine-tuning

# Verificar se é root
if [ "$EUID" -ne 0 ]; then
    echo "Por favor, execute como root ou com sudo"
    exit
fi

# Atualizar pacotes
echo "Atualizando pacotes do sistema..."
apt-get update -qq

# Instalar dependências do sistema
echo "Verificando dependências do sistema..."

# SQLite3 (para avaliação)
if ! command -v sqlite3 &> /dev/null; then
    echo "Instalando sqlite3..."
    apt-get install -y sqlite3
else
    echo "sqlite3 já instalado"
fi

# Python
if ! command -v python3 &> /dev/null; then
    echo "Instalando Python..."
    apt-get install -y python3 python3-pip
else
    echo "Python já instalado"
fi

# Verificar versão do Python (mínimo 3.8)
PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$PYTHON_VERSION" -lt 8 ]; then
    echo "Python 3.8 ou superior é necessário. Versão atual: $(python3 --version)"
    exit 1
fi

# Instalar pip
if ! command -v pip3 &> /dev/null; then
    echo "Instalando pip..."
    apt-get install -y python3-pip
else
    echo "pip já instalado"
fi

# Instalar/verificar pacotes Python
echo "Instalando pacotes Python..."

# Lista de pacotes necessários
PACKAGES=(
    "torch>=2.6.0"
    "torchvision==0.21.0+cu124"
    "torchaudio==2.6.0+cu124"
    "transformers>=4.41.0"
    "datasets>=2.19.0"
    "peft>=0.10.0"
    "trl>=0.8.6"
    "bitsandbytes>=0.43.0"
    "accelerate>=0.29.0"
    "deepeval>=0.17.0"
    "sqlalchemy>=2.0.0"
    "tensorboard>=2.15.0"
    "huggingface-hub>=0.27.0"
)

# Instalar cada pacote se não estiver instalado
for pkg in "${PACKAGES[@]}"; do
    pkg_name=$(echo "$pkg" | cut -d'>' -f1 | cut -d'=' -f1)
    if ! python3 -c "import $pkg_name" &> /dev/null; then
        echo "Instalando $pkg..."
        pip3 install "$pkg"
    else
        echo "$pkg_name já instalado"
    fi
done

# Verificar instalação do CUDA (opcional mas recomendado)
if nvidia-smi &> /dev/null; then
    echo "CUDA/nvidia-smi detectado"
    # Instalar torch com suporte a CUDA se não estiver instalado
    if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        echo "Instalando PyTorch com suporte a CUDA..."
        pip3 install torch --upgrade --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu118
    else
        echo "PyTorch com CUDA já instalado"
    fi
else
    echo "NVIDIA GPU não detectada - instalando PyTorch sem suporte a CUDA"
    pip3 install torch --upgrade
fi

# Configurar git-lfs (para baixar modelos grandes)
if ! command -v git-lfs &> /dev/null; then
    echo "Instalando git-lfs..."
    apt-get install -y git-lfs
    git lfs install
else
    echo "git-lfs já instalado"
fi

# Verificar acesso ao Hugging Face Hub
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "AVISO: Variável HUGGINGFACE_TOKEN não está definida"
    echo "Por favor, defina seu token de acesso do Hugging Face:"
    echo "export HUGGINGFACE_TOKEN='seu_token_aqui'"
    echo "Ou adicione ao seu ~/.bashrc ou ~/.zshrc"
fi

echo "Instalação concluída!"
echo "Para executar o pipeline completo:"
echo "1. projeto/scrpits/python3 preprocess.py"
echo "2. projeto/scrpits/python3 python3 train.py"
echo "3. projeto/scrpits/python3 python3 evaluate.py"