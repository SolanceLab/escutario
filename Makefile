# `make test` runs every test in this room (pytest, using the local .venv when there is one).
PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
.PHONY: test
test:
	$(PY) -m pytest tests -q
