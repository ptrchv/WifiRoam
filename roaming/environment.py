import logging
import math
import pandas as pd
import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from roaming.utils import NetworkConfig, WifiParams, TupleRC
from pathlib import Path
import csv
from collections import deque
import pickle
from roaming.metrics import WifiMetric, WifiStat, METRICS_FN, DsField
import numpy as np
import itertools
import shutil

logger = logging.getLogger(__name__)


@dataclass
class TxInfo:
    acked: bool
    latency: float
    num_tries: int
    rssi: float


@dataclass
class BeaconInfo:
    rssi: float
    snr: float    


class WifiEnvironment(ABC):
    @property
    @abstractmethod
    def map_dims(self) -> TupleRC:
        pass

    @property
    @abstractmethod
    def ap_positions(self) -> list[TupleRC]:
        pass

    @property
    @abstractmethod
    def n_aps(self) -> int:
        pass

    @property
    @abstractmethod
    def metrics(self) -> dict[WifiMetric, set[WifiStat]]:
        pass
    
    @abstractmethod
    def get_metrics(self, sta_pos: TupleRC, ap: int) -> dict[WifiMetric, dict[WifiStat, int|float|None]]:
        pass

    @abstractmethod
    def sample_tx(self, time: float, sta_pos: TupleRC, ap: int) -> TxInfo:
        pass

    @abstractmethod
    def sample_beacons(self, time: float, sta_pos: TupleRC) -> list[BeaconInfo | None]:
        pass


class SimpleWifiEnv(WifiEnvironment):
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
    
    @property
    def metrics(self) -> dict[WifiMetric, set[WifiStat]]:
        return {WifiMetric.RSSI : {WifiStat.MEAN}, WifiMetric.LATENCY: {WifiStat.MEAN}}
    
    def get_metrics(self, sta_pos: TupleRC, ap: int) -> dict[WifiMetric, dict[WifiStat, int|float|None]]:
        rssi = self._calculate_rssi(sta_pos, ap, noise=0)
        latency = self._calculate_latency(rssi, self._net_conf.ap_loads[ap])
        return {WifiMetric.RSSI : {WifiStat.MEAN: rssi}, WifiMetric.LATENCY : {WifiStat.MEAN: latency}}

    def sample_tx(self, time: float, sta_pos: TupleRC, ap: int) -> TxInfo:
        rssi = self._calculate_rssi(sta_pos, ap, noise = np.random.normal(0, 2))
        lat = self._calculate_latency(rssi, self._net_conf.ap_loads[ap])
        return TxInfo(acked=True, latency=lat, num_tries=None, rssi=rssi)
    
    def sample_beacons(self, time: float, sta_pos: TupleRC) -> list[BeaconInfo | None]:
        rssi_list = [self._calculate_rssi(sta_pos, ap, noise = np.random.normal(0, 2)) for ap in range(self.n_aps)]
        return [BeaconInfo(rssi=rssi, snr=None) for rssi in rssi_list]

    def to_json(self) -> str:
        return json.dumps({"net_conf": asdict(self._net_conf), "wifi_params": asdict(self._wifi_params)})

    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return SimpleWifiEnv(NetworkConfig(**data["net_conf"]), WifiParams(**data["wifi_params"]))
    
    def _calculate_rssi(self, sta_pos: TupleRC, ap: int, noise: float) -> float:
        dist = np.linalg.norm(sta_pos.np_array - self._net_conf.ap_positions[ap].np_array)
        path_loss_rssi = -30 - 35 * math.log10(dist + 1)
        return path_loss_rssi + noise

    def _calculate_latency(self, rssi: float, load: float) -> float:
        rssi_quality = 100 + rssi
        latency = 10 + (load * 10) + (150 / (rssi_quality + 5))
        return max(5, latency)

    # def _sample_ap_loads(self):
    #     return {ap: max(0.1, min(1.0, base + np.random.uniform(-0.1, 0.1)))
    #             for ap, base in self._net_conf.base_ap_loads.items()}


class MapWifiEnv:
    def __init__(self, net_conf: NetworkConfig, data_dir: str, cache_dir: str, seed: int, pre_sample: int = 1000):
        self._net_conf = net_conf
        self._data_dir = Path(data_dir)
        self._cache_path = Path(cache_dir) / "MapWifiEnv"
        self._datasets = {}
        self._ap_dataset = []
        self._pre_sample = pre_sample
        self._seed = seed

    @property
    def ap_positions(self):
        return self._net_conf.ap_positions

    @property
    def n_aps(self):
        return len(self._net_conf.ap_positions)

    @property
    def map_dims(self):
        return self._net_conf.map_dims
    
    @property
    def metrics(self):
        return {metric: {stat for stat in stats} for metric, stats in METRICS_FN.items()}

    def load_datasets(self, datasets: list[tuple[str, str]], pre_sample = 1000, use_cache=True):
        self._load_info(datasets)
        if use_cache:
            self._load_cached()
        elif self._cache_path.exists():
            logger.info("Removing cached datasets")
            shutil.rmtree(self._cache_path)
        self._populate_cache()                

    def get_metrics(self, sta_pos: TupleRC, ap: int) -> dict[WifiMetric, dict[WifiStat, int|float|None]]:
        ds_cell = self._pos_to_ds_cell(sta_pos, ap)
        if not ds_cell:
            return None
        pos_metrics = {}
        ds_data, cell_pos = ds_cell
        for metric_full, m_map in ds_data["metric_maps"].items():
            metric, stat = metric_full
            if metric not in pos_metrics:
                pos_metrics[metric] = {}
            val = m_map[cell_pos.row, cell_pos.col]
            pos_metrics[metric][stat] = float(val) if not np.isnan(val) else None
        return pos_metrics

    def sample_tx(self, time: float, sta_pos: TupleRC, ap: int) -> TxInfo | None:
        ds_cell = self._pos_to_ds_cell(sta_pos, ap)
        if ds_cell is None:
            return None
        ds_data, cell_pos = ds_cell
        samples = ds_data["sample_map"][cell_pos.row][cell_pos.col]
        if samples is None:
            return None
        sample = samples.popleft()
        return TxInfo(
            acked=sample[int(DsField.acked)],
            latency=sample[int(DsField.latency)],
            num_tries=sample[int(DsField.transmissions)],
            rssi = sample[int(DsField.rssi)]
        )

    def sample_beacons(self, time: float, sta_pos: TupleRC) -> list[BeaconInfo | None]:
        beacon_list = []
        for ap in range(self.n_aps):
            ds_cell = self._pos_to_ds_cell(sta_pos, ap)
            if ds_cell is not None:
                ds_data, cell_pos = ds_cell
                rssi = ds_data["metric_maps"][(WifiMetric.RSSI, WifiStat.MEAN)][cell_pos.row, cell_pos.col]
                snr = ds_data["metric_maps"][(WifiMetric.SNR, WifiStat.MEAN)][cell_pos.row, cell_pos.col]
                if not np.isnan(rssi) and not np.isnan(snr):
                    beacon_list.append(BeaconInfo(rssi, snr))
                else:
                    beacon_list.append(None)
            else:
                beacon_list.append(None)
        return beacon_list
    
    def _load_info(self, datasets: list[tuple[str, str]]):
        for sim_name, ds_name in datasets:
            if (sim_name, ds_name) not in self._datasets:
                path = self._data_dir / sim_name / "datasets" / ds_name
                with open(path / "info.json") as f:
                    info = json.load(f)
                fmap = []            
                with open(path / "file_map.csv") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        fmap.append([c if c else None for c in row])
                self._datasets[(sim_name, ds_name)] = {"info": info, "fmap": fmap, "sample_map": None, "metric_maps" : {}}
            self._ap_dataset.append((sim_name, ds_name))

    def _load_cached(self):
        for ds_name_full, ds_data in self._datasets.items():
            sim_name, ds_name = ds_name_full
            ds_path = Path(self._cache_path / sim_name / ds_name)
            map_path = ds_path / "maps"
            if map_path.exists():
                for f in map_path.iterdir():
                    fname_list = f.name.split(".")[0].split("-")
                    metric, stat = WifiMetric[fname_list[0]], WifiStat[fname_list[1]]
                    ds_data["metric_maps"][(metric, stat)] = np.genfromtxt(f, delimiter=',')
            sample_path = ds_path / "sample_map.pkl"
            if sample_path.exists():
                with open(sample_path, 'rb') as f:
                    ds_data["sample_map"] = pickle.load(f)

    def _populate_cache(self):
        m_stats = [[(metric, stat) for stat in stats] for metric, stats in METRICS_FN.items()]
        m_stats = set(itertools.chain.from_iterable(m_stats))
        
        for ds_name_full, ds_data in self._datasets.items():
            logger.info("Caching dataset: {}".format(ds_name_full))
            fmap = ds_data["fmap"]
            info = ds_data["info"]
            metric_maps = ds_data["metric_maps"]

            missing_m_stats = m_stats.difference(set(metric_maps.keys()))
            for m_stat in missing_m_stats:
                m_map = np.empty((info["shape"][0], info["shape"][1]))
                m_map[:] = np.nan
                metric_maps[m_stat]= m_map

            sampling = False
            if ds_data["sample_map"] is None:
                sampling = True
                rows, cols = info["shape"][0], info["shape"][1]
                ds_data["sample_map"] = [[None]*cols for _ in range(rows)]            

            for row, file_names in enumerate(fmap):
                for col, fname in enumerate(file_names):
                    self._populate_maps_pos(ds_name_full, TupleRC(row, col), fname, missing_m_stats, sampling)        

            sim_name, ds_name = ds_name_full
            ds_path = self._cache_path / sim_name / ds_name
            map_path = ds_path / "maps"
            map_path.mkdir(exist_ok=True, parents=True)

            for m_stat in missing_m_stats:
                metric, stat = m_stat
                m_map = metric_maps[m_stat]
                np.savetxt(map_path / "{}-{}.csv".format(metric.name, stat.name), m_map, delimiter=",")
            
            if sampling:
                with open(ds_path / "sample_map.pkl", 'wb') as f:
                    pickle.dump(ds_data["sample_map"], f)

    def _populate_maps_pos(self, ds_name_full, pos, fname, missing_m_stats, sampling):
        if not missing_m_stats and not sampling:
            return

        df = None
        if fname is not None:
            sim_name, ds_name = ds_name_full
            df = pd.read_csv(self._data_dir / sim_name / "datasets" / ds_name / "data" / fname)
            if df.empty:
                df = None

        for m_stat in missing_m_stats:
            m_map = self._datasets[ds_name_full]["metric_maps"][m_stat]
            if df is None:
                m_map[pos.row,pos.col] = None
            else:
                metric, stat = m_stat
                m_map[pos.row,pos.col] = METRICS_FN[metric][stat](df)

        if sampling:
            if df is not None:
                samples = df.sample(self._pre_sample, random_state=self._seed)
                samples = deque(samples.itertuples(index=False, name=None))
                self._datasets[ds_name_full]["sample_map"][pos.row][pos.col] = samples

    def _pos_to_ds_cell(self, pos: TupleRC, ap: int):
        ds_name_full = self._ap_dataset[ap]
        ds_data = self._datasets[ds_name_full]
        ap_pos = self._net_conf.ap_positions[ap]
        
        ap_pos_map = TupleRC(ds_data["info"]["ap_pos"][0], ds_data["info"]["ap_pos"][1])
        step = TupleRC(ds_data["info"]["step"][0], ds_data["info"]["step"][1])
        map_origin = TupleRC(ds_data["info"]["range"][0][0], ds_data["info"]["range"][1][0])
        shape = TupleRC(ds_data["info"]["shape"][0], ds_data["info"]["shape"][1])

        cell_pos = pos-ap_pos+ap_pos_map-map_origin
        cell_pos.row = round(cell_pos.row / step.row)
        cell_pos.col = round(cell_pos.col / step.col)

        if 0 < cell_pos.row < shape.row and 0 < cell_pos.col < shape.col:
            return ds_data, cell_pos
        return None
