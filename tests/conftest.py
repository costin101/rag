import os

import pytest

from tests.gating import integration_decision, live_decision, nomad_decision


def pytest_collection_modifyitems(items):
    # The unit lane runs with --disable-socket; an integration test without the opt-out would
    # fail confusingly, so refuse it at collection.
    for item in items:
        if item.get_closest_marker("integration") and not item.get_closest_marker("enable_socket"):
            raise pytest.UsageError(f"{item.nodeid}: integration tests must carry enable_socket")


def pytest_runtest_setup(item):
    if item.get_closest_marker("live"):
        verdict, reason = live_decision(os.environ)
        if verdict != "run":
            pytest.skip(reason if verdict == "skip" else None)
        return
    if not item.get_closest_marker("integration"):
        return
    if item.get_closest_marker("nomad"):
        verdict, reason = nomad_decision(os.environ)
        if verdict == "skip":
            pytest.skip(reason)
        if verdict == "fail":
            pytest.fail(reason, pytrace=False)
    verdict, reason = integration_decision(os.environ)
    if verdict == "skip":
        pytest.skip(reason)
    if verdict == "fail":
        pytest.fail(reason, pytrace=False)
