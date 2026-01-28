import csv
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict, fields
from typing import Callable
import logging
import simpy
import numpy as np
from roaming.utils import TupleRC, SimConfig
from roaming.environment import WifiEnvironment, TxInfo
from roaming.utils import TupleRC
from roaming.roaming import RoamingAlgorithm

logger = logging.getLogger(__name__)

class TrajectorySimulator:
    @dataclass
    class TrajEntry:
        time: int
        segment: int
        ap: int
        count: int
        state: str
        row: float
        col: float
        acked: bool
        rssi: float
        latency: float
        num_tries: float

    @dataclass
    class SimState:
        time: float = 0
        pos: TupleRC =  None
        direction = None
        distance = 0        
        segment: int = -1        
        dataset: list = field(default_factory=list)

    def __init__(self, env: simpy.Environment, wifi_sim: WifiEnvironment, roam_alg: RoamingAlgorithm, sim_config: SimConfig, cache_dir: str, exp_name: str):
        self._env = env
        self._wifi_sim = wifi_sim
        self._roam_alg = roam_alg
        self._config = sim_config
        self._state = None        
        self._trajectory = []
        self._cache_dir =  Path(cache_dir) / "experiments" / exp_name
        self._traj_change_callbacks = []
        self._traj_num = None
        self._generate_trajectory()
        self._load_traj_num()
        self._configure()

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def sim_config(self):
        return self._config
    
    @property
    def traj_num(self) -> int | None:
        return self._traj_num
    
    def register_traj_change_callback(self, callback: Callable[[TupleRC, tuple[TupleRC, TupleRC]], None]):
        self._traj_change_callbacks.append(callback)
    
    def _generate_trajectory(self):
        self._trajectory = [self._gen_pos() for _ in range(self._config.trajectory_len + 1)]

    def _load_traj_num(self):
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        traj_nums = [int(f.name.split("_")[0]) for f in self._cache_dir.iterdir() if f.is_file()]
        self._traj_num = max(traj_nums) + 1 if traj_nums else 0

    def _configure(self):
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
        yield self._env.timeout(self._config.tx_start_time)
        while self._update_position():
            tx_info = TxInfo(acked=False, latency=None, num_tries=None, rssi=None)
            if self._roam_alg.connected:
                tx_sample = self._wifi_sim.sample_tx(self._state.time, self._state.pos, self._roam_alg.ap)
                tx_info = tx_sample if tx_sample is not None else tx_info
            self._state.dataset.append(
                TrajectorySimulator.TrajEntry (
                    time = self._env.now,
                    segment = self._state.segment,
                    ap = self._roam_alg.ap,
                    count = self._roam_alg.count,
                    state = self._roam_alg.state.name,
                    row =  self._state.pos.row,
                    col = self._state.pos.col,
                    acked = tx_info.acked,
                    rssi = tx_info.rssi,
                    latency= tx_info.latency,
                    num_tries = tx_info.num_tries
                )
            )
            self._roam_alg.notify_tx(self._state.pos, tx_info)
            yield self._env.timeout(self._config.pkt_period)
        
        with open(self._cache_dir / "{:02d}_trajectory.csv".format(self._traj_num), "w") as f:
            writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(TrajectorySimulator.TrajEntry)])
            writer.writeheader()
            writer.writerows(asdict(entry) for entry in self._state.dataset)
            
    def _simulate_beacons(self):
        t_out = np.random.uniform(low=0.0, high=self._config.beacon_time)
        yield self._env.timeout(t_out)
        while self._update_position():
            beacon_info = self._wifi_sim.sample_beacons(self._state.time, self._state.pos)
            self._roam_alg.notify_beacon(self._state.pos, beacon_info)
            yield self._env.timeout(self._config.beacon_time)

    def _update_position(self):
        step_len = (self._env.now - self._state.time) * self._config.speed
        segment_change = False
        while step_len >= self._state.distance:
            segment_change = True
            step_len = step_len - self._state.distance
            self._state.segment+=1
            if self._state.segment >= len(self._trajectory) - 1:
                return False                     
            prev, next = self._get_bounds()
            self._state.distance = (next - prev).norm()
            self._state.direction = (next - prev) / self._state.distance
            self._state.pos = prev            
            logger.info("[T={:.6f}s] Segment {} ({:.3f}m) - {} --> {}".format(self._env.now, self._state.segment, self._state.distance, prev, next))
        self._state.pos = self._state.direction * step_len + self._state.pos
        self._state.time = self._env.now
        self._state.distance = self._state.distance - step_len
        if segment_change:
            self._invoke_traj_change_callbacks()
        return True

    def _get_bounds(self):
        return self._trajectory[self._state.segment], self._trajectory[self._state.segment+1]

    def _gen_pos(self):
        dims = self._wifi_sim.map_dims
        return TupleRC(random.randint(0, dims.row - 1), random.randint(0, dims.col - 1))

    def _invoke_traj_change_callbacks(self):
        for callback in self._traj_change_callbacks:
            callback(self._state.pos, self._get_bounds())
