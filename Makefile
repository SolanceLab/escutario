# `make install` installs the Python packages (requirements.txt, then basic-pitch with --no-deps); `make test` runs every test in this room (pytest, using the local .venv when there is one).
PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
.PHONY: install test
install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-deps basic-pitch==0.4.0
test:
	$(PY) -m pytest tests -q
