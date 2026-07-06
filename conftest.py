"""pytest hooks: suppress verbose logging and isolate model artifacts during test runs."""
import logging
import os
import tempfile


def pytest_configure():
    logging.disable(logging.CRITICAL)
    # Point src.config.MODELS_DIR at a throwaway directory *before* any test
    # module imports src.config, so training/evaluation done by tests never
    # reads or clobbers real production artifacts in the project's models/ dir
    # (and vice versa, e.g. a real `python -m src.train` run in progress).
    os.environ.setdefault("MODELS_DIR", tempfile.mkdtemp(prefix="lrp-test-models-"))
