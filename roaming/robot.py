from dataclasses import dataclass
import numpy as np
import math

@dataclass
class TupleRC:
    r: int
    c: int

    @property
    def np_array(self):
        return np.array([self.r, self.c])

@dataclass
class WifiParams:
    rssi_threshold: float = -85.0
    handover_penalty: int = 500
    switch_penalty: int = 50
    no_ap_penalty: int = 2000

MAP_DIMS = TupleXY(120, 60)
WIFI_PARAMS = WifiParams()

AP_POSITIONS = [
    (0, 0),                  # Top-Left (x, y)
    (MAP_DIMS[], 0),         # Top-Right
    (0, MAP_DIMS.x),         # Bottom-Left
    (MAP_DIMS.x, MAP_DIMS.x) # Bottom-Right
]

AP_LOADS = [0.4, 0.4, 0.4, 0.4]

INPUT_FILENAME = 'wifi_signal_dataset.csv'
SIMULATION_SEED = 2
LOAD_SAVED_MODEL = True
MODEL_FILENAME = 'robot_trajectory_model_corrected.keras'


class WifiEnvironment:

    def __init__(self, map_dims: TupleXY = MAP_DIMS, ap_positions: dict[str, tuple] = AP_POSITIONS, ap_params: WifiParams = WIFI_PARAMS, ap_loads: list[float] = AP_LOADS):
        self._map_size = map_dims
        self._ap_params = ap_params
        self._ap_positions = ap_positions
        self._ap_loads = ap_loads
        self._n_aps = len(ap_positions)

        self.n_zones = map_dims.x * map_dims.y

    def calculate_rssi(self, sta_pos: TupleXY, ap_id: int) -> float:
        dist = np.linalg.norm(sta_pos.np_array  - self._ap_positions[ap_id].np_array)
        path_loss_rssi = -30 - 35 * math.log10(dist + 1)
        noise = np.random.normal(0, 2)
        return path_loss_rssi + noise

    def calculate_latency(self, rssi: float, load: float) -> float:
        rssi_quality = 100 + rssi
        latency = 10 + (load * 100) + (150 / (rssi_quality + 5))
        return max(5, latency)
    
    def _sample_ap_loads(self):
        return {ap_id: max(0.1, min(1.0, base + np.random.uniform(-0.1, 0.1)))
                for ap_id, base in self.base_ap_loads.items()}
    
    def _get_coords_from_zone_id(self, zone_id):
        row, col = zone_id // self.map_size, zone_id % self.map_size
        x, y = (col + 0.5) * self.cell_size, (row + 0.5) * self.cell_size
        return np.array([x, y])
    
    def get_map(self, resolution: float = 1.0, samples = 1000):
        map_rows, map_cols = round(self._map_size.y / resolution), round(self._map_size.x/resolution)
        mat_rssi = np.zeros((self.n_aps, map_rows, map_cols))
        mat_rssi = np.zeros((self.n_aps, map_rows, map_cols))
        
        for row in range(0, map_rows):
            for col in range(0, map_cols):
                for i in range(samples):
                    self.calculate_rssi(TupleXY(x=row, y=col), ) 


    # plot_color_gradients('Sequential',
    #                  ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #                   'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #                   'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])


    # def plot_map(self):
    #     mask = (my_data[:, :] < 0)# set -1 to nan
    #     my_data[mask] = np.nan
    #     my_data = my_data * METRICS[metric]["scaling"] # scale to microseconds for latency

    #     ap_pos = []
    #     for ap in configs[conf_num]["config"]["apNodes"]:
    #         pos = (ap["position"]["x"]-x_min)/step, (ap["position"]["y"]-y_min)/step
    #         ap_pos.append(pos)

    #     interf_pos = []
    #     for interf in configs[conf_num]["config"]["interfererNodes"]:
    #         pos = (interf["position"]["x"]-x_min)/step, (interf["position"]["y"]-y_min)/step
    #         interf_pos.append(pos)

    #     print(ap_pos)
    #     print(interf_pos)
    #     fig, ax = plt.subplots()
    #     im = ax.imshow(my_data, cmap="Blues_r")

    #     cbar = ax.figure.colorbar(im, ax=ax)
    #     cbar.ax.set_ylabel(METRICS[metric]["label"], rotation=-90, va="bottom")

    #     #ax.scatter([24], [35], color = "r")
    #     ax.scatter([p[1] for p in ap_pos], [p[0] for p in ap_pos], color = "red")
    #     ax.scatter([p[1] for p in interf_pos], [p[0] for p in interf_pos], color = "orange")        

    #     # # Show all ticks and label them with the respective list entries
    #     # ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    #     # ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)

    #     # # Rotate the tick labels and set their alignment.
    #     # plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
    #     #         rotation_mode="anchor")

    #     # # Loop over data dimensions and create text annotations.
    #     # for i in range(len(y_labels)):
    #     #     for j in range(len(x_labels)):
    #     #         text = ax.text(j, i, my_data[i, j],
    #     #                     ha="center", va="center", color="w")

    #     #ax.set_title("Latency map")
    #     fig.tight_layout()
    #     #plt.show()

    #     fname = plot_dir / metric / "map_{:02}.png".format(conf_num)
    #     fig.savefig(fname)
    #     plt.close(fig)
                
                



def main():
    pass


if __name__ == "__main__":
    main()



# disegnare delle mappe e capire le dimensioni e copertura
