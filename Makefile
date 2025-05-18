VERSION = 0.1.1

# Default target
.DEFAULT_GOAL := help

# Help message generator
.PHONY: help
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
	pixi run python -m build

deploy: build
	python3.11 -m twine upload --repository pypi dist/*

setup:
	pixi install
	python3.11 -m pip install -e .

format:
	pixi run black .
	pixi run ruff .

clean:  ## Remove build artifacts
	rm -rf dist build *.egg-info

