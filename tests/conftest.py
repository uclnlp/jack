import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "sentihood" in item.nodeid:
            item.add_marker(pytest.mark.sentihood)
        elif "SNLI" in item.nodeid:
            item.add_marker(pytest.mark.SNLI)

        if "overfit" in item.nodeid:
            item.add_marker(pytest.mark.overfit)
        elif "smalldata" in item.nodeid:
            item.add_marker(pytest.mark.smalldata)
        elif "readme" in item.nodeid:
            item.add_marker(pytest.mark.readme)
