"""Optional per-UE HetNet observations; legacy QoS-only inputs remain the default."""
import math
from telecom.radio import path_loss_db
import torch
import torch.nn.functional as F


def observation_mode(opt):
    mode = getattr(opt, 'observation_mode', None) or 'qos'
    if mode not in ('qos', 'enriched'):
        raise ValueError(f'Unknown observation_mode: {mode}')
    return mode


def enriched_input_dim(opt, scenario):
    """QoS(N), distance(B), received power(B), serving BS(B), channel(K), SINR(1)."""
    if hasattr(scenario, 'begin_step'):
        raise ValueError('Enriched observations currently support HetNet only')
    return int(opt.nagents) + 3 * scenario.BS_Number() + int(scenario.sce.nChannel) + 1


def enriched_observations(agents, scenario, qos, actions=None, capacity=None):
    """Return per-agent vectors, or None for unchanged legacy observations.

    Geometry/potential received power use current positions. Association, channel,
    global QoS and SINR describe the last completed allocation. No extra environment
    step, reward evaluation, coverage mask, or learning operation is performed.
    SINR is recovered from the simulator's Shannon capacity, avoiding changes to
    the reward API. Missing initial feedback uses zero one-hots and minimum SINR.
    """
    opt = agents[0].opt
    if observation_mode(opt) == 'qos':
        return None
    enriched_input_dim(opt, scenario)
    device = qos.device
    bs = scenario.Get_BaseStations()
    positions = torch.stack([a.location.to(device=device, dtype=torch.float32) for a in agents])
    locations = torch.tensor([b.Get_Location().tolist() for b in bs], device=device, dtype=torch.float32)
    radii = torch.tensor([b.radius for b in bs], device=device, dtype=torch.float32)
    if bool((radii <= 0).any()):
        raise ValueError('Coverage radii must be positive')
    distances = torch.linalg.vector_norm(positions[:, None, :] - locations[None, :, :], dim=-1)
    # Match RewardCalculator's path loss, 1m distance floor and coverage convention.
    radio_distances = distances.clamp_min(1.)
    tx = torch.tensor([b.Transmit_Power_dBm() for b in bs], device=device)
    femto = torch.tensor([b.bs_type == 'FBS' for b in bs], device=device)
    loss = path_loss_db(radio_distances, femto)
    rx_dbm = torch.where(radio_distances <= radii, tx - loss, -140.)
    power_features = (rx_dbm.clamp(-140., 20.) + 140.) / 160.
    n, k = len(agents), int(scenario.sce.nChannel)
    bs_features = torch.zeros((n, len(bs)), device=device)
    channel_features = torch.zeros((n, k), device=device)
    sinr_features = torch.zeros((n, 1), device=device)
    if actions is not None:
        actions = actions.to(device=device, dtype=torch.long).reshape(n)
        bs_features = F.one_hot(actions // k, num_classes=len(bs)).float()
        channel_features = F.one_hot(actions % k, num_classes=k).float()
        if capacity is not None:
            # C = BW log2(1 + SINR). Clamp before expm1 to prevent overflow;
            # anything above 30dB has the same clipped observation anyway.
            exponent = (capacity.to(device) * 1e6 / float(scenario.sce.BW) * math.log(2.)).clamp(0., math.log1p(1000.))
            sinr_db = 10. * torch.expm1(exponent).clamp_min(.01).log10()
            sinr_features = ((sinr_db.clamp(-20., 30.) + 20.) / 50.).reshape(n, 1)
    vectors = torch.cat((qos.float().reshape(1, n).expand(n, n), distances / radii,
                         power_features, bs_features, channel_features, sinr_features), dim=1)
    return {i: vectors[i].detach() for i in range(n)}
