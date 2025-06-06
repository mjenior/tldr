VERSION = "0.4.42"

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

build: format
	pixi lock
	pixi run python -m build

install:
	pixi install
	pip install -e .

clean: 
	rm -rf dist build *.egg-info

update: clean patch build install
