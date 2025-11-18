import math
import numpy as np
import json
from dataclasses import asdict
from roaming.utils import NetworkConfig, WifiParams, TupleRC


class WifiSimulator:
    def __init__(self, net_conf: NetworkConfig, wifi_params: WifiParams):
        self._wifi_params = wifi_params
        self._net_conf = net_conf

    @property
    def map_dims(self):
        return self._net_conf.map_dims

    @property
    def ap_positions(self):
        return self._net_conf.ap_positions

    @property
    def n_aps(self):
        return len(self._net_conf.ap_positions)

    def calculate_rssi(self, sta_pos: TupleRC, ap_id: int) -> float:
        dist = np.linalg.norm(
            sta_pos.np_array - self._net_conf.ap_positions[ap_id].np_array)
        path_loss_rssi = -30 - 35 * math.log10(dist + 1)
        noise = np.random.normal(0, 2)
        return path_loss_rssi + noise

    def calculate_latency(self, rssi: float, load: float) -> float:
        rssi_quality = 100 + rssi
        latency = 10 + (load * 10) + (150 / (rssi_quality + 5))
        return max(5, latency)

    def _sample_ap_loads(self):
        return {ap_id: max(0.1, min(1.0, base + np.random.uniform(-0.1, 0.1)))
                for ap_id, base in self._net_conf.base_ap_loads.items()}

    def sample_tx(self, time: float | None, sta_pos: TupleRC, ap: int) -> tuple[float, float]:
        rssi = self.calculate_rssi(sta_pos, ap)
        lat = self.calculate_latency(rssi, self._net_conf.ap_loads[ap])
        return rssi, lat
    
    def sample_beacons(self, time: float, sta_pos) -> list[float]:
        return [self.calculate_rssi(sta_pos, ap) for ap in range(self.n_aps)]

    def to_json(self) -> str:
        return json.dumps({"net_conf": asdict(self._net_conf), "wifi_params": asdict(self._wifi_params)})

    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return WifiSimulator(NetworkConfig(**data["net_conf"]), WifiParams(**data["wifi_params"]))
