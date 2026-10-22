from pathlib import Path

from graph2mat._slow_tests import mark_slow


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: above 1 s in a full run; skip with -m 'not slow'")


def pytest_collection_modifyitems(items):
    mark_slow(items, Path(__file__).parent, prefix="tests/")
