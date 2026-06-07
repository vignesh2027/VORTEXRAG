.PHONY: test install demo benchmark clean test-fast test-tve test-vrc test-rfg test-ccb test-fv test-sdc test-cpg test-integration test-e2e

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install -e .

install-full:
	pip install -e .
	pip install sentence-transformers>=2.2.0
	pip install spacy>=3.5.0
	python -m spacy download en_core_web_sm

install-demo:
	pip install gradio>=4.0 pandas>=1.5.0

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v -m "not slow" --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=core --cov=vortexrag --cov-report=term-missing

# Individual layer tests
test-tve:
	pytest tests/test_tve.py -v --tb=short

test-vrc:
	pytest tests/test_vrc.py -v --tb=short

test-rfg:
	pytest tests/test_rfg.py -v --tb=short

test-ccb:
	pytest tests/test_ccb.py -v --tb=short

test-fv:
	pytest tests/test_fv.py -v --tb=short

test-sdc:
	pytest tests/test_sdc.py -v --tb=short

test-cpg:
	pytest tests/test_cpg.py -v --tb=short

test-integration:
	pytest tests/test_integration.py -v --tb=short

test-e2e:
	pytest tests/test_e2e.py -v --tb=short

# ── Examples ──────────────────────────────────────────────────────────────────

demo:
	python examples/demo_gradio.py --cli

demo-gradio:
	python examples/demo_gradio.py --port 7860

benchmark:
	python examples/benchmark_eval.py --mock --n 5 --all

benchmark-full:
	python examples/benchmark_eval.py --all --n 50 --output results.csv --markdown results.md

domains:
	python examples/domain_examples.py --domain all

# ── Paper compilation ─────────────────────────────────────────────────────────

paper:
	cd paper && bash compile.sh

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage 2>/dev/null || true

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "VORTEXRAG Makefile"
	@echo "=================="
	@echo ""
	@echo "Installation:"
	@echo "  make install          Install package in editable mode"
	@echo "  make install-full     Install with optional sentence-transformers + spaCy"
	@echo "  make install-demo     Install Gradio + pandas for demo"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run full test suite"
	@echo "  make test-fast        Run tests excluding @pytest.mark.slow"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make test-tve         Run TVE tests only"
	@echo "  make test-vrc         Run VRC tests only"
	@echo "  make test-sdc         Run SDC tests only"
	@echo "  make test-cpg         Run CPG tests only"
	@echo "  make test-rfg         Run RFG tests only"
	@echo "  make test-ccb         Run CCB tests only"
	@echo "  make test-fv          Run FV tests only"
	@echo "  make test-integration Run integration tests"
	@echo "  make test-e2e         Run end-to-end tests"
	@echo ""
	@echo "Examples:"
	@echo "  make demo             Run CLI demo (no Gradio needed)"
	@echo "  make demo-gradio      Launch Gradio web demo on port 7860"
	@echo "  make benchmark        Quick benchmark (mock data, 5 samples)"
	@echo "  make benchmark-full   Full benchmark (50 samples, saves CSV)"
	@echo "  make domains          Run all 11 domain examples"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove __pycache__, .pyc, .pytest_cache"
	@echo ""
