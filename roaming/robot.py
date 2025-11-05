import json
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
from roaming.utils import TupleRC, NetworkConfig, WifiParams
from dataclasses import asdict, dataclass, field
import random
import pandas as pd


DATA_FOLDER = "data"
EXP_NAME = "test3"
SIMULATION_SEED = 2

MAP_DIMS = TupleRC(60, 120)
NET_CONFIG = NetworkConfig(
    map_dims = MAP_DIMS,
    ap_positions = [
        TupleRC(0, 0),                              # Top-Left
        TupleRC(MAP_DIMS.row-1, 0),                 # Bottom-Left
        TupleRC(0, MAP_DIMS.col-1),                 # Top-Right
        TupleRC(MAP_DIMS.row-1, MAP_DIMS.col-1),    # Bottom-Right
        TupleRC(MAP_DIMS.row//2, MAP_DIMS.col//2)   # Center
    ],
    ap_loads= [0.2, 0.2, 0.2, 0.2, 0.6]
)

WIFI_PARAMS = WifiParams (
    rssi_threshold = -85.0,
    handover_penalty = 500,
    switch_penalty = 50,
    no_ap_penalty = 2000,
)


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

    def sample_oracle(self, sta_pos: TupleRC, ap: int) -> tuple[float, float]:
        rssi = self.calculate_rssi(sta_pos, ap)
        lat = self.calculate_latency(rssi, self._net_conf.ap_loads[ap])
        return rssi, lat

    def to_json(self) -> str:
        return json.dumps({"net_conf": asdict(self._net_conf), "wifi_params": asdict(self._wifi_params)})

    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return WifiSimulator(NetworkConfig(**data["net_conf"]), WifiParams(**data["wifi_params"]))


class MapPlotter:
    def __init__(self, data_dir: str, exp_name: str):
        self._exp_dir = Path(data_dir) / exp_name
        self._exp_dir.mkdir(parents=True, exist_ok=True)

        self._wifi_sim = None
        self._mat_rssi = None
        self._mat_lat = None

    @property
    def map_loaded(self):
        return self._mat_rssi is not None and self._mat_lat is not None
    
    def set_simulator(self, wifi_sim: WifiSimulator):
        self._wifi_sim = wifi_sim

    def load_from_file(self):
        conf_file = self._exp_dir / "config.json"
        if conf_file.exists():
            with open(conf_file, "r") as f:
                self._wifi_sim = WifiSimulator.from_json(f.read())
            rssi_map_file = self._exp_dir / "maps" / "rssi.npy"
            lat_map_file = self._exp_dir / "maps" / "lat.npy"
            if rssi_map_file.exists() and lat_map_file.exists():
                self._mat_rssi = np.load(rssi_map_file)
                self._mat_lat = np.load(lat_map_file)
            return True
        return False

    def generate_maps(self, resolution: float = 1.0, num_samples = 10, save=True):
        map_rows, map_cols = round(
            self._wifi_sim.map_dims.row / resolution), round(self._wifi_sim.map_dims.col/resolution)
        self._mat_rssi = np.zeros((self._wifi_sim.n_aps, map_rows, map_cols))
        self._mat_lat = np.zeros((self._wifi_sim.n_aps, map_rows, map_cols))
        for ap in range(self._wifi_sim.n_aps):
            print("Simulating AP {} in {}".format(ap,self._wifi_sim.ap_positions[ap]))
            for row in range(0, map_rows):
                for col in range(0, map_cols):
                    sample_rssi, sample_latency = [], []
                    for _ in range(num_samples):
                        rssi, lat = self._wifi_sim.sample_oracle(TupleRC(row, col), ap)
                        sample_rssi.append(rssi)
                        sample_latency.append(lat)
                    self._mat_rssi[ap, row, col] = np.mean(sample_rssi)
                    self._mat_lat[ap, row, col] = np.mean(sample_latency)
        if save:
            with open(self._exp_dir / "config.json", "w") as f:
                f.write(self._wifi_sim.to_json())
            map_dir = (self._exp_dir / "maps")
            map_dir.mkdir(parents=True, exist_ok=True)
            np.save(map_dir / "rssi.npy", self._mat_rssi, allow_pickle=False)
            np.save(map_dir / "lat.npy", self._mat_lat, allow_pickle=False)

    def plot_maps(self, extension: str = "png"):
        if self._mat_rssi is None or self._mat_lat is None:
            raise ValueError("Maps not generated yet.")
        
        plot_dir = (self._exp_dir / "plots")
        plot_dir.mkdir(parents=True, exist_ok=True)
    
        mat_rssi_best = np.max(self._mat_rssi, axis=0)
        mat_lat_best = np.min(self._mat_lat, axis=0)

        mat_lat_multi = self._mat_lat.copy()
        for ap in range(self._wifi_sim.n_aps):
            mask = (mat_lat_multi[ap, :, :] > mat_lat_best[:, :])
            mat_lat_multi[ap, mask] = np.nan

        mat_rssi_multi = self._mat_rssi.copy()
        for ap in range(self._wifi_sim.n_aps):
            mask = (mat_rssi_multi[ap, :, :] < mat_rssi_best[:, :])
            mat_rssi_multi[ap, mask] = np.nan

        fig, ax = plt.subplots()
        im = ax.imshow(mat_rssi_best[:, :])
        fig.tight_layout()
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("rssi", rotation=-90, va="bottom")
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        fig.savefig(plot_dir / "rssi.{}".format(extension))
        plt.close(fig)

        fig, ax = plt.subplots()
        im = ax.imshow(mat_lat_best[:, :], cmap="Blues_r")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("latency", rotation=-90, va="bottom")
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        fig.tight_layout()
        fig.savefig(plot_dir / "latency.{}".format(extension))
        plt.close(fig)

        cmaps_r = [plt.cm.Greys_r, plt.cm.Purples_r, plt.cm.Blues_r, plt.cm.Greens_r, plt.cm.Oranges_r, plt.cm.Reds_r]
        cmaps = [plt.cm.Greys, plt.cm.Purples, plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Reds]
        
        fig, ax = plt.subplots()
        min_lat = np.nanmin(mat_lat_multi)
        max_lat = np.nanmax(mat_lat_multi)
        for ap in range(self._wifi_sim.n_aps):
            im = ax.imshow(mat_lat_multi[ap, :, :], cmap=cmaps_r[ap], vmin=min_lat, vmax=max_lat)
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        fig.tight_layout()
        fig.savefig(plot_dir / "latency_multi.{}".format(extension))
        plt.close(fig)
         
        fig, ax = plt.subplots()
        min_rssi = np.nanmin(mat_rssi_multi)
        max_rssi = np.nanmax(mat_rssi_multi)
        for ap in range(self._wifi_sim.n_aps):            
            im = ax.imshow(mat_rssi_multi[ap, :, :], cmap=cmaps[ap], vmin=min_rssi, vmax=max_rssi)
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        fig.tight_layout()
        fig.savefig(plot_dir / "rssi_multi.{}".format(extension))
        plt.close(fig)


class RoamingAlgorithm:
    pass


class TrajectorySimulator:
    @dataclass
    class SimState:
        period: float
        speed: float
        ap: int
        pos: TupleRC =  None
        segment: int = -1
        segment_points: list[TupleRC] = field(default_factory=list)
        residual: float = 0

    def __init__(self, wifi_sim: WifiSimulator, alg: RoamingAlgorithm):
        self._wifi_sim = wifi_sim
        self._alg = alg
        self._state = None
        
        self._trajectory = []
        self._dataset = pd.DataFrame(columns=["segment", "x_pos", "y_pos", "ap", "rssi", "latency"])

    @property
    def trajectory(self):
        return self._trajectory
    
    def generate_trajectory(self, num_segments: int):
        self._trajectory = [self._gen_pos() for _ in range(num_segments + 1)]

    def simulate(self, period: float = 0.01, speed: float = 0.5):
        if not self._trajectory:
            raise ValueError("Trajectory not generated yet.")
    
        # print(self._trajectory)

        # Find AP with greater RSSI
        best_ap, best_rss = None, None
        for ap in range(self._wifi_sim.n_aps):
            rssi, _ = self._wifi_sim.sample_oracle(self._trajectory[0], ap)
            if best_rss is None or rssi > best_rss:
                best_ap, best_rss = ap, rssi

        # Setup initial state
        self._state = TrajectorySimulator.SimState(speed=speed, period=period, ap=best_ap)             

        while self.move_fwd():
            rssi, lat = self._wifi_sim.sample_oracle(self._state.pos, self._state.ap)
            self._dataset.loc[len(self._dataset)] = [
                self._state.segment, self._state.pos.row, self._state.pos.col,
                self._state.ap, rssi, lat]
        print(self._dataset.shape[0])
        self._dataset.to_csv("data/trajectory.csv")
            

    def move_fwd(self):
        if not self._state.segment_points:
            self._state.segment += 1
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

        step_len = (self._state.period * self._state.speed)
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


def main():
    # map_plt = MapPlotter(data_dir=DATA_FOLDER, exp_name=EXP_NAME)
    # if not map_plt.load_from_file():
    #     wifi_sim = WifiSimulator(net_conf=NET_CONFIG, wifi_params=WIFI_PARAMS)
    #     map_plt.set_simulator(wifi_sim)
    # if not map_plt.map_loaded:
    #     map_plt.generate_maps()        
    # map_plt.plot_maps()

    wifi_sim = WifiSimulator(net_conf=NET_CONFIG, wifi_params=WIFI_PARAMS)
    traj_sim = TrajectorySimulator(wifi_sim=wifi_sim, alg=RoamingAlgorithm())
    traj_sim.generate_trajectory(3)
    traj_sim.simulate(period=0.01, speed=0.5)

if __name__ == "__main__":
    random.seed(SIMULATION_SEED)
    np.random.seed(SIMULATION_SEED)
    main()


# TODO list
# Bisogna rivedere l'algoritmo di movimento
# - se l'applicazione genera pacchetto e viene accodato, le condizioni di tramissione sono quelle di quando fa la tramissione non di quando lo genera).
# - però come faccio a considerare segmenti indipendenti?
# - magari mettere tra variabili la lunghezza della coda pacchetti.
# - i punti di tramissione non sono più equidistanti

# finish trajectory simulator
# fare codice che croppa matrici quando sono sotto rssi minimo
# controllare che AP siano nel posto giusto
# disegnare delle mappe e capire le dimensioni e copertura
# fare simulatore traiettorie
# simulare in ns-3 una mappa su cui fare esperimenti reali
# add type hints to functions
# change TupleRC to float
# plot trajectories 
