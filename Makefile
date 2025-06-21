VERSION = "1.0.32"

# Default target
.DEFAULT_GOAL := help

# Help message generator
help: 
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# -----------------------------------------------------------------------------

.PHONY: patch minor major
patch minor major:
	bumpversion --current-version ${VERSION} $@ --allow-dirty --no-tag --no-commit

format:
	pixi run black tldr/*.py
	pixi run black ui.py

build: format
	pixi lock
	pixi run python -m build

install: build
	pixi install
	pixi run pip install -e .
	rm -rf dist build tldr.egg-info
