from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from roaming.metrics import WifiMetric, WifiStat
from roaming.utils import TupleRC
from roaming.environment import WifiEnvironment


class MapPlotter:
    COLORS = ["black", "purple", "blue", "green", "orange", "red"]    
    CMAPS = [plt.cm.Greys, plt.cm.Purples, plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Reds]
    CMAPS_R = [plt.cm.Greys_r, plt.cm.Purples_r, plt.cm.Blues_r, plt.cm.Greens_r, plt.cm.Oranges_r, plt.cm.Reds_r]

    def __init__(self, cache_dir: str, exp_name: str):
        self._cache_dir = Path(cache_dir) / exp_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._wifi_sim = None
        self._mat_rssi = None
        self._mat_lat = None

    @property
    def map_loaded(self):
        return self._mat_rssi is not None and self._mat_lat is not None
    
    def set_simulator(self, wifi_sim: WifiEnvironment):
        self._wifi_sim = wifi_sim

    def load_from_file(self):
        conf_file = self._cache_dir / "config.json"
        if conf_file.exists():
            with open(conf_file, "r") as f:
                self._wifi_sim = WifiEnvironment.from_json(f.read())
            rssi_map_file = self._cache_dir / "maps" / "rssi.npy"
            lat_map_file = self._cache_dir / "maps" / "lat.npy"
            if rssi_map_file.exists() and lat_map_file.exists():
                self._mat_rssi = np.load(rssi_map_file)
                self._mat_lat = np.load(lat_map_file)
            return True
        return False

    def generate_maps(self, resolution: float = 1.0, save=True):
        map_rows, map_cols = round(
            self._wifi_sim.map_dims.row / resolution), round(self._wifi_sim.map_dims.col/resolution)
        self._mat_rssi = np.empty((self._wifi_sim.n_aps, map_rows, map_cols))
        self._mat_lat = np.empty((self._wifi_sim.n_aps, map_rows, map_cols))
        self._mat_rssi[:] = np.nan
        self._mat_lat[:] = np.nan
        for ap in range(self._wifi_sim.n_aps):
            print("Simulating AP {} in {}".format(ap,self._wifi_sim.ap_positions[ap]))
            for row in range(0, map_rows):
                for col in range(0, map_cols):
                    metrics = self._wifi_sim.get_metrics(sta_pos=TupleRC(row, col), ap=ap)
                    self._mat_rssi[ap, row, col] = metrics[WifiMetric.RSSI][WifiStat.MEAN]
                    self._mat_lat[ap, row, col] = metrics[WifiMetric.LATENCY][WifiStat.MEAN]

        if save:
            with open(self._cache_dir / "config.json", "w") as f:
                f.write(self._wifi_sim.to_json())
            map_dir = (self._cache_dir / "maps")
            map_dir.mkdir(parents=True, exist_ok=True)
            np.save(map_dir / "rssi.npy", self._mat_rssi, allow_pickle=False)
            np.save(map_dir / "lat.npy", self._mat_lat, allow_pickle=False)

    def plot_maps(self, extension: str = "png", trajectory=False):
        if self._mat_rssi is None or self._mat_lat is None:
            raise ValueError("Maps not generated yet.")
        fig_dict = {}
        
        plot_dir = (self._cache_dir / "plots")
        plot_dir.mkdir(parents=True, exist_ok=True)
    
        mat_rssi_best = np.nanmax(self._mat_rssi, axis=0)
        mat_lat_best = np.nanmin(self._mat_lat, axis=0)

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
        fig_dict["rssi"] = fig, ax

        fig, ax = plt.subplots()
        im = ax.imshow(mat_lat_best[:, :], cmap="Blues_r")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("latency", rotation=-90, va="bottom")
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        fig.tight_layout()
        fig.savefig(plot_dir / "latency.{}".format(extension))
        fig_dict["latency"] = fig, ax
        
        fig, ax = plt.subplots()
        min_lat = np.nanmin(mat_lat_multi)
        max_lat = np.nanmax(mat_lat_multi)
        for ap in range(self._wifi_sim.n_aps):
            im = ax.imshow(mat_lat_multi[ap, :, :], cmap=MapPlotter.CMAPS_R[ap], vmin=min_lat, vmax=max_lat)
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        fig.tight_layout()
        fig.savefig(plot_dir / "latency_multi.{}".format(extension))
        fig_dict["latency_multy"] = fig, ax
         
        fig, ax = plt.subplots()
        min_rssi = np.nanmin(mat_rssi_multi)
        max_rssi = np.nanmax(mat_rssi_multi)
        for ap in range(self._wifi_sim.n_aps):            
            im = ax.imshow(mat_rssi_multi[ap, :, :], cmap=MapPlotter.CMAPS[ap], vmin=min_rssi, vmax=max_rssi)
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        fig.tight_layout()
        fig.savefig(plot_dir / "rssi_multi.{}".format(extension))
        fig_dict["rssi_multi"] = fig , ax

        if trajectory:
            df_trj = pd.read_csv(self._cache_dir / "trajectory.csv")
            segments = df_trj.groupby(["segment", "ap", "count"]).agg(
                time=('time', lambda row: row.iloc[0]),
                start_row=('row', lambda row: row.iloc[0]), start_col=('col', lambda row: row.iloc[0]),
                end_row=('row', lambda row: row.iloc[-1]), end_col=('col', lambda row: row.iloc[-1])).sort_values(by='time', ascending=True).reset_index()

            plt.ion()
            for row in segments.itertuples():
                print("Segment {} - {} - {} --> {}".format(row.segment, row.ap, TupleRC(row=row.start_row, col=row.start_col), TupleRC(row=row.end_row, col=row.end_col)))
                for fig, ax in fig_dict.values():
                    ax.plot([row.start_col, row.end_col], [row.start_row, row.end_row], color=MapPlotter.COLORS[int(row.ap)])
                plt.pause(0.01)
                input()

        for fig, _ in fig_dict.values():
            plt.close(fig)
