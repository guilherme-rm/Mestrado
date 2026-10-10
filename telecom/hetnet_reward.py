"""HetNet joint-allocation reward and radio evaluation."""
from typing import Tuple
import numpy as np
import torch
from functions.configuration import config_value
from .radio import received_power_mw

class RewardCalculator:
    """Encapsulates reward computation logic with caching for performance."""
    
    def __init__(self, sce, opt, device: torch.device):
        self.sce = sce
        self.opt = opt
        self.device = device
        # Caches for performance - populated on first use
        self._bs_locations_np = None  # (nBS, 2) numpy array
        self._bs_locations_t = None   # (nBS, 2) torch tensor
        self._bs_tx_powers_t = None   # (nBS,) tx power in dBm
        self._bs_radii_t = None       # (nBS,) coverage radius
        self._bs_type_code_t = None   # (nBS,) 0 for MBS/PBS, 1 for FBS
        self._bs_list_cache = None
    
    def _init_bs_cache(self, BS_list):
        """Initialize BS location cache for vectorized operations."""
        if self._bs_list_cache is not BS_list or self._bs_locations_np is None:
            self._bs_list_cache = BS_list
            self._bs_locations_np = np.array([bs.Get_Location() for bs in BS_list], dtype=np.float32)
            self._bs_locations_t = torch.as_tensor(
                self._bs_locations_np, device=self.device, dtype=torch.float32
            )
            self._bs_tx_powers_t = torch.tensor(
                [float(bs.Transmit_Power_dBm()) for bs in BS_list],
                device=self.device,
                dtype=torch.float32,
            )
            self._bs_radii_t = torch.tensor(
                [float(getattr(bs, "radius", float("inf"))) for bs in BS_list],
                device=self.device,
                dtype=torch.float32,
            )
            self._bs_type_code_t = torch.tensor(
                [1.0 if getattr(bs, "bs_type", "MBS") == "FBS" else 0.0 for bs in BS_list],
                device=self.device,
                dtype=torch.float32,
            )

    def _rx_power_from_bs_index(self, bs_idx: int, agent_location: torch.Tensor) -> torch.Tensor:
        """Compute received power in mW using torch path-loss on the current device."""
        bs_loc = self._bs_locations_t[bs_idx]
        diff = bs_loc - agent_location
        d = torch.linalg.vector_norm(diff).clamp_min(1.0)

        tx_power_dbm = self._bs_tx_powers_t[bs_idx]
        is_fbs = self._bs_type_code_t[bs_idx]

        return received_power_mw(d, tx_power_dbm, self._bs_radii_t[bs_idx], is_fbs)

    def compute(
        self,
        action_all: torch.Tensor,
        action_i: torch.Tensor,
        agent_location: torch.Tensor,
        scenario,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Compute QoS satisfaction, reward, and capacity.
        
        Reward = profit * Rate - power_cost * Tx_power_dBm - action_cost
        (utility-based reward with action selection cost)
        """
        BS_list = scenario.Get_BaseStations()
        K = self.sce.nChannel

        BS_selected = int(action_i // K)
        Ch_selected = int(action_i % K)

        # Basic validation for action index
        if BS_selected < 0 or BS_selected >= len(BS_list):
            return self._invalid_action_result()

        self._init_bs_cache(BS_list)

        rx_power_t = self._rx_power_from_bs_index(BS_selected, agent_location)
        rx_power = float(rx_power_t.detach().cpu())
        if rx_power <= 1e-20:
            return self._invalid_action_result()

        interference = self._compute_interference(
            action_all, Ch_selected, BS_list, agent_location, rx_power
        )
        
        sinr, capacity_mbps = self._compute_sinr_and_capacity(rx_power, interference)
        cap_tensor = torch.tensor(float(capacity_mbps), device=self.device)

        # Compute utility-based reward
        # Rate in Mbps (same as capacity_mbps)
        rate = capacity_mbps
        
        # Get config parameters with defaults
        profit = float(config_value(self.sce, "profit", 0.5))
        power_cost = float(config_value(self.sce, "power_cost", 0.0005))
        action_cost = float(config_value(self.sce, "action_cost", 0.001))
        
        # Get transmit power of selected BS
        tx_power_dbm = float(self._bs_tx_powers_t[BS_selected].detach().cpu())
        
        # Calculate reward: utility (profit * rate) minus costs
        reward_val = profit * rate - power_cost * tx_power_dbm - action_cost
        
        # Determine QoS satisfaction (binary)
        if sinr >= 10 ** (self.sce.QoS_thr / 10):
            qos = 1
        else:
            qos = 0

        return (qos, torch.tensor(reward_val, device=self.device, dtype=torch.float32), cap_tensor)
    
    def _invalid_action_result(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        return (
            0,
            torch.tensor(self.sce.negative_cost, device=self.device, dtype=torch.float32),
            torch.tensor(0.0, device=self.device),
        )
    
    def _compute_rx_power(self, bs, agent_location: torch.Tensor) -> float:
        BS_location = torch.tensor(bs.Get_Location(), device=self.device, dtype=torch.float32)
        Loc_diff = BS_location - agent_location
        distance = torch.sqrt(Loc_diff[0] ** 2 + Loc_diff[1] ** 2)
        d_val = max(float(distance.item()), 1.0)
        return bs.Receive_Power(d_val)
    
    def _compute_interference(
        self, action_all, channel, BS_list, agent_location, rx_power
    ) -> float:
        """Compute interference using vectorized torch operations on device."""
        K = self.sce.nChannel
        
        # Initialize BS cache if needed
        self._init_bs_cache(BS_list)
        
        # Keep actions on device to avoid host-device transfers.
        if isinstance(action_all, torch.Tensor):
            actions_t = action_all.to(self.device)
        else:
            actions_t = torch.as_tensor(action_all, device=self.device)
        actions_t = actions_t.to(torch.long).view(-1)
        
        # Vectorized channel and BS extraction
        channels = actions_t % K
        bs_indices = actions_t // K
        
        # Find agents on the same channel (excluding those with invalid BS)
        same_channel_mask = (channels == int(channel)) & (bs_indices >= 0) & (bs_indices < len(BS_list))
        
        if not bool(same_channel_mask.any()):
            return 0.0
        
        # Keep location on device
        if isinstance(agent_location, torch.Tensor):
            loc_t = agent_location.to(self.device, dtype=torch.float32)
        else:
            loc_t = torch.as_tensor(agent_location, device=self.device, dtype=torch.float32)
        
        # Get BS indices and locations for interfering agents
        interfering_bs_idx = bs_indices[same_channel_mask]
        interfering_bs_locs = self._bs_locations_t[interfering_bs_idx]
        
        # Vectorized distance calculation
        loc_diff = interfering_bs_locs - loc_t.unsqueeze(0)  # (n_interfering, 2)
        distances = torch.linalg.vector_norm(loc_diff, dim=1).clamp_min(1.0)

        # Per-BS pathloss model in torch (MBS/PBS vs FBS)
        tx_dbm = self._bs_tx_powers_t[interfering_bs_idx]
        is_fbs = self._bs_type_code_t[interfering_bs_idx]
        radii = self._bs_radii_t[interfering_bs_idx]

        rx_mw = received_power_mw(distances, tx_dbm, radii, is_fbs)

        interference = float(rx_mw.sum().detach().cpu())
        return max(0.0, interference - rx_power)
    
    def _compute_sinr_and_capacity(self, rx_power: float, interference: float) -> Tuple[float, float]:
        noise = 10 ** (self.sce.N0 / 10) * self.sce.BW
        sinr = rx_power / (interference + noise)
        spec_eff = np.log2(1.0 + sinr) if sinr > 0 else 0.0
        capacity_bps = self.sce.BW * spec_eff
        return sinr, capacity_bps / 1e6

