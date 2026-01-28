import json
from typing import get_args, get_origin
from dataclasses import dataclass, is_dataclass, asdict
from pathlib import Path
import numpy as np
import math


def nested_dataclass(*args, **kwargs):
    def wrapper(cls):
        cls = dataclass(cls, **kwargs)
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            for name, value in kwargs.items():
                field_type = cls.__annotations__.get(name, None)
                # nested single dataclass
                if is_dataclass(field_type) and isinstance(value, dict):
                    new_obj = field_type(**value)
                    kwargs[name] = new_obj
                # nested list of dataclasses, checks if:
                # - the type hint of the field is List
                # - if the type hint says that it contains dataclasses
                # - if the provided default value is a list
                if get_origin(field_type) == list and is_dataclass(get_args(field_type)[0]) and isinstance(value, list):
                    # initializes the list
                    # - copies the provided values if they are the hinted datacasses
                    # - otherwise initizialized the a new dataclass object with the value in the list
                    kwargs[name] = [v if isinstance(v, get_args(field_type)[0]) else get_args(field_type)[0](**v) for v in value]
            original_init(self, *args, **kwargs)
        cls.__init__ = __init__
        return cls
    return wrapper(args[0]) if args else wrapper


@dataclass
class TupleRC:
    row: float
    col: float

    def __add__(self, other):
        return TupleRC(self.row + other.row, self.col + other.col)
    
    def __sub__(self, other):
        return TupleRC(self.row - other.row, self.col - other.col)
    
    def __mul__(self, val):
        return TupleRC(self.row * val, self.col * val)

    def __truediv__(self, val):
        if isinstance(val, (int, float)):
            if val == 0:
                raise ZeroDivisionError("division by zero")
            return TupleRC(self.row / val, self.col / val)
        return NotImplemented
    
    def round(self, digits: int | None = None):
        self.col = round(self.col, digits)
        self.row = round(self.row, digits)
        return self
    
    def norm(self):
        return math.sqrt(self.row **2 + self.col**2)
    
    @property
    def np_array(self):
        return np.array([self.row, self.col])
    

@dataclass
class ExpConfig:
    exp_name: str
    data_dir: str
    cache_dir: str
    sim_seed: int


@nested_dataclass
class NetworkConfig:
    map_dims: TupleRC
    ap_positions: list[TupleRC]
    datasets : list[str]
    ap_loads: list[float]

    def __post_init__(self):
        if self.ap_positions:
            if len(self.ap_positions) != len(self.ap_loads):
                raise ValueError("Number of AP positions and loads must be the same.")
            return
        

@nested_dataclass
class WifiConfig:
    roaming_alg: str
    rssi_threshold: float
    roaming_time: float
    min_switch_time: float | None

        
@dataclass
class SimConfig:
    trajectory_len: int
    pkt_period: float
    speed: float
    beacon_time: float
    tx_start_time: float


@nested_dataclass
class Config:
    exp_conf: ExpConfig
    net_conf: NetworkConfig
    wifi_conf: WifiConfig
    sim_conf: SimConfig


def load_config(fname: str) -> Config:
    with open(fname) as f:
        conf = json.load(f)
    return Config(**conf)


def save_config(conf: Config, traj_num: int):
    exp_conf = conf.exp_conf
    out_dir = Path(exp_conf.cache_dir)/ "experiments"/ exp_conf.exp_name / "confs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / "{:02d}_conf.json".format(traj_num)    
    with open(fname, "w") as f:
        json.dump(asdict(conf), f, indent=4)
