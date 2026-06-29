.PHONY: install run test lint docker clean

install:
	pip install -r requirements.txt

train:
	python train.py

run:
	streamlit run app/main.py

test:
	pytest -q

lint:
	ruff check src app tests

docker:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache data/cache.db