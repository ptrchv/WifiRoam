import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from roaming.metrics import WifiMetric, WifiStat
from roaming.utils import TupleRC
from roaming.environment import WifiEnvironment

logger = logging.getLogger(__name__)


class MapPlotter:
    COLORS = ["black", "purple", "blue", "green", "orange", "red"]    
    CMAPS = [plt.cm.Greys, plt.cm.Purples, plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Reds]
    CMAPS_R = [plt.cm.Greys_r, plt.cm.Purples_r, plt.cm.Blues_r, plt.cm.Greens_r, plt.cm.Oranges_r, plt.cm.Reds_r]

    def __init__(self, wifi_sim: WifiEnvironment, cache_dir: str, exp_name: str):
        self._cache_dir = Path(cache_dir) / "experiments" / exp_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._wifi_sim = wifi_sim
        self._mat_rssi = None
        self._mat_lat = None
        self._mat_num_tries = None

    @property
    def map_loaded(self):
        return self._mat_rssi is not None and self._mat_lat is not None

    def generate_maps(self, resolution: float = 1.0):
        map_rows, map_cols = round(
            self._wifi_sim.map_dims.row / resolution), round(self._wifi_sim.map_dims.col/resolution)
        self._mat_rssi = np.empty((self._wifi_sim.n_aps, map_rows, map_cols))
        self._mat_lat = np.empty((self._wifi_sim.n_aps, map_rows, map_cols))
        self._mat_num_tries = np.empty((self._wifi_sim.n_aps, map_rows, map_cols))        
        self._mat_rssi[:] = np.nan
        self._mat_lat[:] = np.nan
        self._mat_num_tries[:] = np.nan
        for ap in range(self._wifi_sim.n_aps):
            #print("Simulating AP {} in {}".format(ap,self._wifi_sim.ap_positions[ap]))
            for row in range(0, map_rows):
                for col in range(0, map_cols):
                    metrics = self._wifi_sim.get_metrics(sta_pos=TupleRC(row, col), ap=ap)
                    if metrics is None:
                        self._mat_rssi[ap, row, col] = np.nan
                        self._mat_lat[ap, row, col] = np.nan
                        self._mat_num_tries[ap, row, col] = np.nan
                    else:
                        self._mat_rssi[ap, row, col] = metrics[WifiMetric.RSSI][WifiStat.MEAN]
                        self._mat_lat[ap, row, col] = metrics[WifiMetric.LATENCY][WifiStat.MEAN]
                        self._mat_num_tries[ap, row, col] = metrics[WifiMetric.NUM_TRIES][WifiStat.MEAN]

    def plot_maps(self, extension: str = "png", traj_num=None, interactive=False):
        if self._mat_rssi is None or self._mat_lat is None:
            raise ValueError("Maps not generated yet.")
        fig_dict = {}
        
        # create plot dir
        plot_dir = self._cache_dir / "plots"
        plot_dir = plot_dir / "{:02d}".format(traj_num) if traj_num else plot_dir 
        plot_dir.mkdir(parents=True, exist_ok=True)

        # create matrices for best metric value in each point
        mat_rssi_best = np.nanmax(self._mat_rssi, axis=0)
        mat_lat_best = np.nanmin(self._mat_lat, axis=0)
        mat_num_tries_best = np.nanmin(self._mat_num_tries, axis=0)        

        mat_rssi_multi = self._mat_rssi.copy()
        for ap in range(self._wifi_sim.n_aps):
            mask = (mat_rssi_multi[ap, :, :] < mat_rssi_best[:, :])
            mat_rssi_multi[ap, mask] = np.nan

        mat_lat_multi = self._mat_lat.copy()
        for ap in range(self._wifi_sim.n_aps):
            mask = (mat_lat_multi[ap, :, :] > mat_lat_best[:, :])
            mat_lat_multi[ap, mask] = np.nan

        mat_num_tries_multi = self._mat_num_tries.copy()
        for ap in range(self._wifi_sim.n_aps):
            mask = (mat_num_tries_multi[ap, :, :] > mat_num_tries_best[:, :])
            mat_num_tries_multi[ap, mask] = np.nan

        # plot best metric values matrices
        fig, ax = plt.subplots()
        im = ax.imshow(mat_rssi_best[:, :])
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("RSSI (dBm)", rotation=-90, va="bottom")
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        ax.set_xlabel("Y position (m)")
        ax.set_ylabel("X position (m)")
        fig.tight_layout()
        fig_dict["rssi"] = fig, ax

        fig, ax = plt.subplots()
        im = ax.imshow(mat_lat_best[:, :], cmap="Blues_r")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Latency", rotation=-90, va="bottom")
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        ax.set_xlabel("Y position (m)")
        ax.set_ylabel("X position (m)")
        fig.tight_layout()
        fig_dict["latency"] = fig, ax

        fig, ax = plt.subplots()
        im = ax.imshow(mat_num_tries_best[:, :], cmap="Blues_r")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Num. tries (mean)", rotation=-90, va="bottom")
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        ax.scatter([30, 70], [30, 30], c = "orange")
        ax.set_xlabel("Y position (m)")
        ax.set_ylabel("X position (m)")
        fig.tight_layout()
        fig_dict["num_tries"] = fig, ax
        
        # plot multicolor matrices for best metric
        fig, ax = plt.subplots()
        min_rssi = np.nanmin(mat_rssi_multi)
        max_rssi = np.nanmax(mat_rssi_multi)
        for ap in range(self._wifi_sim.n_aps):            
            im = ax.imshow(mat_rssi_multi[ap, :, :], cmap=MapPlotter.CMAPS[ap], vmin=min_rssi, vmax=max_rssi)
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        ax.set_xlabel("Y position (m)")
        ax.set_ylabel("X position (m)")
        fig.tight_layout()
        fig_dict["rssi_multi"] = fig , ax
        
        fig, ax = plt.subplots()
        min_lat = np.nanmin(mat_lat_multi)
        max_lat = np.nanmax(mat_lat_multi)
        for ap in range(self._wifi_sim.n_aps):
            im = ax.imshow(mat_lat_multi[ap, :, :], cmap=MapPlotter.CMAPS_R[ap], vmin=min_lat, vmax=max_lat)
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        ax.set_xlabel("Y position (m)")
        ax.set_ylabel("X position (m)")
        fig.tight_layout()
        fig_dict["latency_multy"] = fig, ax

        fig, ax = plt.subplots()
        min_num_tries = np.nanmin(mat_num_tries_multi)
        max_num_tries = np.nanmax(mat_num_tries_multi)
        for ap in range(self._wifi_sim.n_aps):
            im = ax.imshow(mat_num_tries_multi[ap, :, :], cmap=MapPlotter.CMAPS_R[ap], vmin=min_num_tries, vmax=max_num_tries)
        ax.scatter([pos.col for pos in self._wifi_sim.ap_positions], [pos.row for pos in self._wifi_sim.ap_positions], c = "red")
        ax.set_xlabel("Y position (m)")
        ax.set_ylabel("X position (m)")
        fig.tight_layout()
        fig_dict["num_tries_multi"] = fig, ax

        # save all figures
        for name, img in fig_dict.items():
            fig, _ = img
            fig.savefig(plot_dir / "{}.{}".format(name, extension))

        # plot trajectories
        if traj_num is not None:    
            df_trj = pd.read_csv(self._cache_dir / "{:02d}_trajectory.csv".format(traj_num))
            segments = df_trj.groupby(["segment", "ap", "count"]).agg(
                time=('time', lambda row: row.iloc[0]),
                start_row=('row', lambda row: row.iloc[0]), start_col=('col', lambda row: row.iloc[0]),
                end_row=('row', lambda row: row.iloc[-1]), end_col=('col', lambda row: row.iloc[-1])
            ).sort_values(by='time', ascending=True).reset_index()

            plot_dir_trj = self._cache_dir / "plots_trj" / "{:02d}".format(traj_num)
            plot_dir_trj.mkdir(parents=True, exist_ok=True)
            
            plt.ion()
            for row in segments.itertuples():
                logger.info("Segment {} - {} - {} --> {}".format(row.segment, row.ap, TupleRC(row=row.start_row, col=row.start_col), TupleRC(row=row.end_row, col=row.end_col)))
                for fig, ax in fig_dict.values():
                    ax.plot([row.start_col, row.end_col], [row.start_row, row.end_row], color=MapPlotter.COLORS[int(row.ap)])
                if interactive:
                    plt.pause(0.001)
                    input()

            # save all figures with trajectory
            for name, img in fig_dict.items():
                fig, _ = img
                fig.savefig(plot_dir_trj / "{}.{}".format(name, extension))

        # close all figures
        for fig, _ in fig_dict.values():
            plt.close(fig)
