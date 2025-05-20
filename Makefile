VERSION = "0.2.3"

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

build: format
	pixi lock
	pixi run python -m build

#deploy: build
#	python -m twine upload --repository pypi dist/*

install:
	pixi install
	python3.11 -m pip install -e .

format:
	pixi run black tldr/*.py

clean: 
	rm -rf build *.egg-info

