.PHONY: run install clean check runner
.DEFAULT_GOAL:=runner
run: install
	cd src && poetry run python runner.py

install: pyproject.toml
	poetry install

clean:
	for /d /r %%d in (__pycache__) do rmdir /q /s "%%d"

check:
	poetry run flake8 src/

runner: check run clean