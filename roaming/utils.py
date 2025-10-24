from typing import get_args, get_origin
from dataclasses import dataclass, is_dataclass
import numpy as np

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
    row: int
    col: int

    @property
    def np_array(self):
        return np.array([self.row, self.col])

@nested_dataclass
class WifiParams:
    rssi_threshold: float
    handover_penalty: int
    switch_penalty: int
    no_ap_penalty: int

@nested_dataclass
class NetworkConfig:
    map_dims: TupleRC
    ap_positions: list[TupleRC]     
    ap_loads: list[float]

    def __post_init__(self):
        if self.ap_positions:
            if len(self.ap_positions) != len(self.ap_loads):
                raise ValueError("Number of AP positions and loads must be the same.")
            return