from dataclasses import dataclass, field
import simpy
import pandas as pd
import random

from roaming.environment import WifiSimulator
from roaming.utils import TupleRC
from roaming.roaming import RoamingAlgorithm
    

class TrajectorySimulator:
    @dataclass
    class SimState:
        pos: TupleRC =  None
        segment: int = None
        ap: int = None
        segment_points: list[TupleRC] = field(default_factory=list)
        residual: float = 0
        dataset: pd.DataFrame = None

    @dataclass
    class SimConfig:
        exp_name: str
        period: float
        speed: float

    def __init__(self, env: simpy.Environment, wifi_sim: WifiSimulator, alg: RoamingAlgorithm):
        self._env = env
        self._wifi_sim = wifi_sim
        self._alg = alg
        self._state = None
        self._config = None
        self._trajectory = []

    @property
    def trajectory(self):
        return self._trajectory
    
    def generate_trajectory(self, num_segments: int):
        self._trajectory = [self._gen_pos() for _ in range(num_segments + 1)]

    def configure(self, exp_name: str, period: float = 0.01, speed: float = 0.5):
        self._config = TrajectorySimulator.SimConfig(exp_name=exp_name, period=period, speed=speed)
        self._env.process(self.movement())
    
    def simulate(self):
        if not self._trajectory:
            raise ValueError("Trajectory not generated yet.")
        if self._config is None:
            raise ValueError("Simulation configuration not set yet.")

        # Find AP with greater RSSI
        best_ap, best_rss = None, None
        for ap in range(self._wifi_sim.n_aps):
            rssi, _ = self._wifi_sim.sample_oracle(self._trajectory[0], ap)
            if best_rss is None or rssi > best_rss:
                best_ap, best_rss = ap, rssi

        # Setup initial state
        self._state = TrajectorySimulator.SimState(
            ap=best_ap, segment=-1,
            dataset= pd.DataFrame(columns=["segment", "x_pos", "y_pos", "ap", "rssi", "latency"])
        )

        # Simulation loop
        while self.move_fwd():
            rssi, lat = self._wifi_sim.sample_oracle(self._state.pos, self._state.ap)
            self._state.dataset.loc[len(self._state.dataset)] = [
                self._state.segment, self._state.pos.row, self._state.pos.col,
                self._state.ap, rssi, lat]
            yield self._env.timeout(self._config.period)

        print(self._state.dataset.shape[0])
        self._state.dataset.to_csv("data/{}/trajectory.csv".format(self._config.exp_name), index=False)

    def move_fwd(self):
        if not self._state.segment_points:
            self._state.segment = self._state.segment + 1
            if self._state.segment == len(self._trajectory) - 1:
                return False
            if not self.load_segment():
                return False            
        self._state.pos = self._state.segment_points.pop(0)
        return True
    
    def load_segment(self):
        prev, next = self._get_bounds()
        diff = next - prev
        distance = diff.norm()
        while self._state.residual >= distance:
            self._state.segment += 1
            if self._segment == len(self._trajectory) - 1:
                return False
            prev, next = self._get_bounds()
            diff = next - prev
            distance = diff.norm()
            self._state.residual = self._state.residual - distance

        step_len = (self._config.period * self._config.speed)
        dir_vect = diff / distance

        prev = dir_vect * self._state.residual + prev
        num_steps = round((distance - self._state.residual) // step_len)
        self._state.residual = (num_steps + 1) * step_len - distance
        self._state.segment_points = [prev + dir_vect * step_len * i for i in range(num_steps)]
        print("Segment {} - {} --> {}".format(self._state.segment, self._trajectory[self._state.segment], self._trajectory[self._state.segment+1]))
        return True

    def _get_bounds(self):
        return self._trajectory[self._state.segment], self._trajectory[self._state.segment+1]

    def _gen_pos(self):
        dims = self._wifi_sim.map_dims
        return TupleRC(random.randint(0, dims.col - 1), random.randint(0, dims.row - 1))
