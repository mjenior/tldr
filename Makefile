VERSION = "1.0.30"

# Default target
.DEFAULT_GOAL := help

# Help message generator
help: 
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# -----------------------------------------------------------------------------

patch:
	bumpversion --current-version ${VERSION} patch --allow-dirty --no-tag --no-commit

minor:
	bumpversion --current-version ${VERSION} minor --allow-dirty --no-tag --no-commit

major:
	bumpversion --current-version ${VERSION} major --allow-dirty --no-tag --no-commit

format:
	pixi run black tldr/*.py

build: clean patch
	pixi lock
	pixi run python -m build
	rm -rf build tldr.egg-info

install:
	pixi install
	pip install -e .
	rm -rf build tldr.egg-info

clean: 
	rm -rf dist build tldr.egg-info
