from dataclasses import dataclass
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class TupleRC:
    row: int
    col: int

    @property
    def np_array(self):
        return np.array([self.row, self.col])

@dataclass
class WifiParams:
    rssi_threshold: float = -85.0
    handover_penalty: int = 500
    switch_penalty: int = 50
    no_ap_penalty: int = 2000


MAP_DIMS = TupleRC(row=60, col=120)
WIFI_PARAMS = WifiParams()

AP_POSITIONS = [
    TupleRC(0, 0),                          # Top-Left (x, y)
    TupleRC(MAP_DIMS.row-1, 0),             # Bottom-Left
    TupleRC(0, MAP_DIMS.col-1),             # Top-Right
    TupleRC(MAP_DIMS.row-1, MAP_DIMS.col-1),  # Bottom-Right
    TupleRC(MAP_DIMS.row//2, MAP_DIMS.col//2),  # Center
]

AP_LOADS = [0.2, 0.2, 0.2, 0.2, 0.6]


# SIMULATION_SEED = 2 


class WifiSimulator:
    def __init__(self, map_dims: TupleRC = MAP_DIMS, ap_positions: dict[str, tuple] = AP_POSITIONS, wifi_params: WifiParams = WIFI_PARAMS, ap_loads: list[float] = AP_LOADS):
        self._map_size = map_dims
        self._wifi_params = wifi_params
        self._ap_positions = ap_positions
        self._ap_loads = ap_loads
        self._n_aps = len(ap_positions)

    @property
    def map_size(self):
        return self._map_size

    @property
    def n_aps(self):
        return self._n_aps

    @property
    def ap_positions(self):
        return self._ap_positions

    def calculate_rssi(self, sta_pos: TupleRC, ap_id: int) -> float:
        dist = np.linalg.norm(
            sta_pos.np_array - self._ap_positions[ap_id].np_array)
        path_loss_rssi = -30 - 35 * math.log10(dist + 1)
        noise = np.random.normal(0, 2)
        return path_loss_rssi + noise

    def calculate_latency(self, rssi: float, load: float) -> float:
        rssi_quality = 100 + rssi
        latency = 10 + (load * 10) + (150 / (rssi_quality + 5))
        return max(5, latency)

    def _sample_ap_loads(self):
        return {ap_id: max(0.1, min(1.0, base + np.random.uniform(-0.1, 0.1)))
                for ap_id, base in self.base_ap_loads.items()}

    def sample_oracle(self, sta_pos: TupleRC, ap: int) -> tuple[float, float]:
        rssi = self.calculate_rssi(sta_pos, ap)
        lat = self.calculate_latency(rssi, self._ap_loads[ap])
        return rssi, lat


class MapPlotter:
    def __init__(self):
        self._mat_rssi = None
        self._mat_lat = None
        self._n_aps = None
        self._map_size = None
        self._ap_positions = None

    def load_maps(self, fname: str, ap_positions=AP_POSITIONS):
        self._mat_rssi = np.load("{}_rssi.npy".format(fname))
        self._mat_lat = np.load("{}_lat.npy".format(fname))
        self._map_size = TupleRC(self._mat_rssi.shape[1], self._mat_rssi.shape[2])
        self._n_aps = self._mat_rssi.shape[0]
        self._ap_positions = ap_positions

    def generate_maps(self, wifi_sim: WifiSimulator, resolution: float = 1.0, num_samples = 10, fname=None):
        self._map_size = wifi_sim.map_size
        self._n_aps = wifi_sim.n_aps
        self._ap_positions = wifi_sim.ap_positions

        map_rows, map_cols = round(
            self._map_size.row / resolution), round(self._map_size.col/resolution)
        self._mat_rssi = np.zeros((self._n_aps, map_rows, map_cols))
        self._mat_lat = np.zeros((self._n_aps, map_rows, map_cols))
        for ap in range(self._n_aps):
            print("Simulating AP {} in {}".format(ap,self._ap_positions[ap]))
            for row in range(0, map_rows):
                for col in range(0, map_cols):                
                    sample_rssi, sample_latency = [], []
                    for _ in range(num_samples):
                        rssi, lat = wifi_sim.sample_oracle(TupleRC(row, col), ap)
                        sample_rssi.append(rssi)
                        sample_latency.append(lat)
                    self._mat_rssi[ap, row, col] = np.mean(sample_rssi)
                    self._mat_lat[ap, row, col] = np.mean(sample_latency)
        if fname:
            np.save("{}_rssi.npy".format(fname), self._mat_rssi, allow_pickle=False)
            np.save("{}_lat.npy".format(fname), self._mat_lat, allow_pickle=False)


    def plot_maps(self, fname: str, extension: str = "png"):
        if self._mat_rssi is None or self._mat_lat is None:
            raise ValueError("Maps not generated yet.")
    
        mat_rssi_best = np.max(self._mat_rssi, axis=0)
        mat_lat_best = np.min(self._mat_lat, axis=0)

        mat_lat_multi = self._mat_lat.copy()
        for ap in range(self._n_aps):
            mask = (mat_lat_multi[ap, :, :] > mat_lat_best[:, :])
            mat_lat_multi[ap, mask] = np.nan

        mat_rssi_multi = self._mat_rssi.copy()
        for ap in range(self._n_aps):
            mask = (mat_rssi_multi[ap, :, :] < mat_rssi_best[:, :])
            mat_rssi_multi[ap, mask] = np.nan  

        fig, ax = plt.subplots()
        im = ax.imshow(mat_rssi_best[:, :])
        fig.tight_layout()
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("rssi", rotation=-90, va="bottom")
        ax.scatter([pos.col for pos in self._ap_positions], [pos.row for pos in self._ap_positions], c = "red")
        fig.savefig("{}_rssi.{}".format(fname, extension))
        plt.close(fig)

        fig, ax = plt.subplots()
        im = ax.imshow(mat_lat_best[:, :], cmap="Blues_r")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("latency", rotation=-90, va="bottom")
        ax.scatter([pos.col for pos in self._ap_positions], [pos.row for pos in self._ap_positions], c = "red")
        fig.tight_layout()
        fig.savefig("{}_latency.{}".format(fname, extension))
        plt.close(fig)

        cmaps_r = [plt.cm.Greys_r, plt.cm.Purples_r, plt.cm.Blues_r, plt.cm.Greens_r, plt.cm.Oranges_r, plt.cm.Reds_r]
        cmaps = [plt.cm.Greys, plt.cm.Purples, plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Reds]        
        
        fig, ax = plt.subplots()
        min_lat = np.nanmin(mat_lat_multi)
        max_lat = np.nanmax(mat_lat_multi)
        for ap in range(self._n_aps):            
            im = ax.imshow(mat_lat_multi[ap, :, :], cmap=cmaps_r[ap], vmin=min_lat, vmax=max_lat)
        ax.scatter([pos.col for pos in self._ap_positions], [pos.row for pos in self._ap_positions], c = "red")
        fig.tight_layout()
        fig.savefig("{}_latency_multi.{}".format(fname, extension))
        plt.close(fig)        
        
        fig, ax = plt.subplots()
        min_rssi = np.nanmin(mat_rssi_multi)
        max_rssi = np.nanmax(mat_rssi_multi)
        for ap in range(self._n_aps):            
            im = ax.imshow(mat_rssi_multi[ap, :, :], cmap=cmaps[ap], vmin=min_rssi, vmax=max_rssi)
        ax.scatter([pos.col for pos in self._ap_positions], [pos.row for pos in self._ap_positions], c = "red")
        fig.tight_layout()
        fig.savefig("{}_rssi_multi.{}".format(fname, extension))
        plt.close(fig)


def main():
    EXP_NAME = "test3"

    wifi_sim = WifiSimulator()
    map_plt = MapPlotter()    
    #map_plt.generate_maps(wifi_sim, fname="test3")

    map_plt.load_maps(EXP_NAME)
    map_plt.plot_maps(EXP_NAME)


if __name__ == "__main__":
    main()

# fare codice che croppa matrici quando sono sotto rssi minimo
# controllare che AP siano nel posto giusto
# disegnare delle mappe e capire le dimensioni e copertura
# separate files (maps and plots) in folders
# move network configuration to data structure
# when saving also put configuration file for simulation
# fare simulatore traiettorie
# simulare in ns-3 una mappa su cui fare esperimenti reali
