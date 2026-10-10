"""Shared HetNet path loss: metres in, dBm/dB or linear mW out."""
import math
import torch


def path_loss_db(distance_m, is_femto):
    if torch.is_tensor(distance_m):
        distance_m = distance_m.clamp_min(1.)
        femto = torch.as_tensor(is_femto, device=distance_m.device).to(distance_m.dtype)
        return (34. + 40. * distance_m.log10()) * (1. - femto) + (37. + 30. * distance_m.log10()) * femto
    distance_m = max(float(distance_m), 1.)
    return (37. + 30. * math.log10(distance_m)) if is_femto else (34. + 40. * math.log10(distance_m))


def received_power_mw(distance_m, tx_dbm, radius_m, is_femto):
    if torch.is_tensor(distance_m):
        d = distance_m.clamp_min(1.)
        power = torch.pow(10., (tx_dbm - path_loss_db(d, is_femto)) / 10.)
        return torch.where(d <= radius_m, power, torch.zeros_like(power))
    d = max(float(distance_m), 1.)
    return 10 ** ((tx_dbm - path_loss_db(d, is_femto)) / 10.) if d <= radius_m else 0.
