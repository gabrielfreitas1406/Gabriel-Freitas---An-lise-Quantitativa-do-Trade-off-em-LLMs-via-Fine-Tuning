# Makefile para projeto SQL LLM Fine-tuning

.PHONY: install check_env train evaluate all

VENV_NAME?=venv
PYTHON=${VENV_NAME}/bin/python3

all: install preprocess train evaluate

install:
    @echo "Instalando dependências..."
    apt-get update -qq && apt-get install -y python3 python3-pip sqlite3 git-lfs
    python3 -m pip install --upgrade pip
    python3 -m pip install torch transformers datasets peft trl bitsandbytes accelerate deepeval sqlalchemy tensorboard huggingface-hub
    git lfs install
    @echo "\nInstalação concluída. Configure seu token do Hugging Face com:"
    @echo "export HUGGINGFACE_TOKEN='seu_token_aqui'"

check_env:
    @echo "Verificando ambiente..."
    @python3 -c "import sys; assert sys.version_info >= (3,8), 'Python 3.8+ necessário'; print('Python OK')"
    @python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA não disponível'; print('CUDA OK')" || echo "CUDA não disponível - treinamento será lento"
    @test -n "$$HUGGINGFACE_TOKEN" || (echo "HUGGINGFACE_TOKEN não definido" && exit 1)

preprocess:
    @echo "Pré-processando dados..."
    python3 preprocess.py

train:
    @echo "Iniciando treinamento..."
    python3 train.py

evaluate:
    @echo "Avaliando modelo..."
    python3 evaluate.py

clean:
    @echo "Limpando arquivos temporários..."
    rm -rf __pycache__ processed_data/* results/*