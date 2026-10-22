# hu-demand-forecasting — reproducible tasks (uv + make)
# Usage: `make setup` then `make notebook`. Run `make help` to list targets.

.DEFAULT_GOAL := help
UV := uv

.PHONY: help setup pipeline notebook figures clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the environment and install the package (uv)
	$(UV) sync --extra notebook
	$(UV) pip install -e .

pipeline: ## Run the end-to-end pipeline (ingestion -> baseline) in the terminal
	$(UV) run python -c "import huforecast as hf, json; a=hf.run_pipeline(); \
print('raw', a['raw'].shape, '| clean', a['clean'].shape); \
print(json.dumps(a['baselines'], indent=2, default=str, ensure_ascii=False))"

notebook: ## Execute the audit notebook end to end and save outputs
	$(UV) run jupyter nbconvert --to notebook --execute --inplace notebooks/auditoria_baseline.ipynb

figures: ## (optional) regenerate EDA figures into docs/figuras
	$(UV) run python -c "from huforecast.reporting import make_figures; make_figures()"

clean: ## Remove generated processed data and caches
	rm -f data/processed/*.parquet
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
