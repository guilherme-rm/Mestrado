from rl.interaction import movement_due


def test_movement_cadence_matches_completed_decision_boundaries():
    assert movement_due(5, 5, 100)
    assert not movement_due(4, 5, 100)
    assert not movement_due(0, 5, 100)
    assert not movement_due(100, 5, 100)
    assert not movement_due(5, 0, 100)