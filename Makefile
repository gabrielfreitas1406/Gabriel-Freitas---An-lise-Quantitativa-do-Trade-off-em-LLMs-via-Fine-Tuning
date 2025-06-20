.PHONY: help venv install clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make venv      - Cria ambiente virtual"
	@echo "  make install   - Instala dependências"
	@echo "  make clean     - Remove ambiente virtual"
	@echo "  make setup     - Cria venv e instala dependências (tudo em um passo)"

venv:
	@echo "Criando ambiente virtual..."
	python -m venv llm_tradeoff
	@echo "Ambiente virtual criado. Ative com:"
	@echo "  source llm_tradeoff/bin/activate  # Linux/MacOS"
	@echo "  ou llm_tradeoff\\Scripts\\activate  # Windows"

install:
	@echo "Instalando dependências..."
	./llm_tradeoff/bin/pip install --upgrade pip
	./llm_tradeoff/bin/pip install torch transformers datasets peft accelerate sqlite3 deepeval==0.21.x pytest
	@echo "Dependências instaladas com sucesso!"

setup: venv install
	@echo "Ambiente configurado e pronto para uso!"

clean:
	@echo "Removendo ambiente virtual..."
	rm -rf llm_tradeoff
	@echo "Ambiente virtual removido"