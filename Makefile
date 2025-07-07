VERSION = "1.2.0"

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

build: format
	pixi lock
	pixi run python -m build

install: build 
	pixi install
	pixi run pip install -e .

clean:
	rm -rf dist build tldr.egg-info
	git clean -fdx -e .venv/ -e .pixi/ -e '*.pyc' -e '*.pyo' -e '__pycache__'

all: format build install clean

uninstall:
	pip uninstall tldr -y
	pixi uninstall

publish: build
	pixi publish
	git tag ${VERSION}
	git push origin ${VERSION}
	git push origin --tags
