"""Canonical per-step policy metrics."""


def qos_summary(qos):
    """Summarize per-user QoS values for one joint allocation."""
    values = qos.reshape(-1)
    if values.numel() == 0:
        raise ValueError('QoS metrics require at least one user')
    satisfied = values.eq(1)
    return {
        'qos_fraction': float(values.float().mean().item()),
        'satisfied_users': int(satisfied.sum().item()),
        'total_users': int(values.numel()),
        'all_satisfied': bool(satisfied.all().item()),
    }