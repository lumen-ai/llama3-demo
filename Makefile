.PHONY upgrade:
upgrade:
	pip install --upgrade transformers
	pip install accelerate
	pip install python-dotenv
	
	
.PHONY run:
run:
	python main.py
