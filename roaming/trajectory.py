from dataclasses import dataclass, field
import simpy
import pandas as pd
import random

from roaming.environment import WifiSimulator
from roaming.utils import TupleRC
from roaming.roaming import RoamingAlgorithm


@dataclass
class SimConfig:
    exp_name: str
    pkt_period: float
    speed: float
    beacon_time: float = 0.1


class TrajectorySimulator:
    @dataclass
    class SimState:
        time: float = 0
        pos: TupleRC =  None
        direction = None
        distance = 0        
        segment: int = -1        
        segment_points: list[TupleRC] = field(default_factory=list)
        dataset: pd.DataFrame = field(default_factory= lambda: pd.DataFrame(columns=["time", "segment", "ap", "count", "row", "col", "rssi", "latency"]))

    def __init__(self, env: simpy.Environment, wifi_sim: WifiSimulator, roam_alg: RoamingAlgorithm):
        self._env = env
        self._wifi_sim = wifi_sim
        self._roam_alg = roam_alg
        self._state = None
        self._config = None
        self._trajectory = []

    @property
    def trajectory(self):
        return self._trajectory
    
    def generate_trajectory(self, num_segments: int):
        self._trajectory = [self._gen_pos() for _ in range(num_segments + 1)]

    def configure(self, sim_config: SimConfig):
        self._config = sim_config
        self._setup_sim()
        self._env.process(self._simulate_traffic())
        self._env.process(self._simulate_beacons())

    def _setup_sim(self):
        if not self._trajectory:
            raise ValueError("Trajectory not generated yet.")
        if self._config is None:
            raise ValueError("Simulation configuration not set yet.")
        
        # Setup initial state
        self._state = TrajectorySimulator.SimState()
    
    def _simulate_traffic(self):
        # Simulation loop
        while self._update_position():
            if self._roam_alg.connected:
                rssi, lat = self._wifi_sim.sample_tx(self._state.time, self._state.pos, self._roam_alg.ap)
                self._roam_alg.notify_tx(self._state.pos, rssi, lat)
            else:
                rssi, lat = None, None
            self._state.dataset.loc[len(self._state.dataset)] = [
                    self._env.now, self._state.segment, self._roam_alg.ap, self._roam_alg.count,
                    self._state.pos.row, self._state.pos.col, rssi, lat]
            yield self._env.timeout(self._config.pkt_period)

        print(self._state.dataset.shape[0])
        self._state.dataset.to_csv("data/{}/trajectory.csv".format(self._config.exp_name), index=False)

    def _simulate_beacons(self):
        while self._update_position():
            rssi_list = self._wifi_sim.sample_beacons(self._state.time, self._state.pos)
            self._roam_alg.notify_beacon(self._state.pos, rssi_list)
            yield self._env.timeout(self._config.beacon_time)

    def _update_position(self):
        step_len = (self._env.now - self._state.time) * self._config.speed
        while step_len >= self._state.distance:
            step_len = step_len - self._state.distance
            self._state.segment+=1
            if self._state.segment >= len(self._trajectory) - 1:
                return False            
            prev, next = self._get_bounds()
            self._state.distance = (next - prev).norm()
            self._state.direction = (next - prev) / self._state.distance
            self._state.pos = prev
            print("[T={:.6f}s] Segment {} ({:.3f}m) - {} --> {}".format(self._env.now, self._state.segment, self._state.distance, prev, next))
        self._state.pos = self._state.direction * step_len + self._state.pos
        self._state.time = self._env.now
        self._state.distance = self._state.distance - step_len
        return True

    def _get_bounds(self):
        return self._trajectory[self._state.segment], self._trajectory[self._state.segment+1]

    def _gen_pos(self):
        dims = self._wifi_sim.map_dims
        return TupleRC(random.randint(0, dims.row - 1), random.randint(0, dims.col - 1))
