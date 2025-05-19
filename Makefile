VERSION = 0.1.1

# Default target
.DEFAULT_GOAL := help

# Help message generator
help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# -----------------------------------------------------------------------------

patch:
	#bumpversion --current-version ${VERSION} patch --allow-dirty --no-tag --no-commit
	bump2version patch

bump-minor:
	bump2version minor

bump-major:
	bump2version major

build: format
	pixi lock
	pixi run python -m build

deploy: build
	python -m twine upload --repository pypi dist/*

install:
	pixi install
	pip install -e .

format:
	pixi run black tldr/*.py

clean: 
	rm -rf build *.egg-info

