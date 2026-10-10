import pytest
import torch

from rl.metrics import qos_summary


def test_qos_summary_preserves_soft_rate_and_counts_satisfied_users():
    summary = qos_summary(torch.tensor([1.0, 0.5, 0.0]))

    assert summary == {
        'qos_fraction': pytest.approx(0.5),
        'satisfied_users': 1,
        'total_users': 3,
        'all_satisfied': False,
    }


def test_qos_summary_handles_binary_qos():
    summary = qos_summary(torch.tensor([1, 1, 0]))

    assert summary['qos_fraction'] == pytest.approx(2 / 3)
    assert summary['all_satisfied'] is False