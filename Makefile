make run:
	uvicorn main:app --reload

make lint:
	ruff check --fix
	ruff format