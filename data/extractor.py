#%%
import json
from pathlib import Path
import copy
import csv

BASE_DIR = Path("./handover_hi_res")
EXP_DIR = BASE_DIR / "raw"

EXP_FILE_SUFFIX = ".dat"
EXTRACT_EXP_NUM = lambda x: int(x.name.split(".")[0].replace("db", ""))
# EXTRACT_EXP_NUM = lambda x: int(x.name.split(".")[0].split("_")[1])

EXP_FILE =  BASE_DIR / "experiments.json"                    # maps experiment file -> configuration
CONF_FILE = BASE_DIR / "configs.json"                        # single experiments grouped by network configuration
DATASET_DIR = BASE_DIR / "datasets"

FEATURES = {
    "seq": lambda r: r["seq"],
    "acked": lambda r: r["acked"],
    "latency": lambda r: r["latency"],
    "transmissions": lambda r: len(r["transmissions"]),
    "rssi": lambda r: r["transmissions"][0]["rssi"],
    "noise": lambda r: r["transmissions"][0]["noise"]
}


def extract_headers(exp_dir, fname):
    exp_dir = Path(exp_dir)
    fname = Path(fname)

    paths = sorted([p for p in exp_dir.iterdir() if p.is_file() and p.suffix == EXP_FILE_SUFFIX])
    paths = sorted(paths, key=EXTRACT_EXP_NUM)

    conf_dict = {}
    errors = []
    for p in paths:
        with p.open() as f:       
            try:        
                line = f.readlines(2)[1].strip()[:-1]
                content = json.loads(line)
                conf_dict[p.name] = content
                print(p.name) 
            except Exception:
                conf_dict[p.name] = None
                errors.append(p)
                print("error: {}".format(p.name))

    print([p.name for p in errors])
    print([k for k, v in conf_dict.items() if v is None])

    fname.parent.absolute().mkdir(parents=True, exist_ok=True)
    with open(fname, "w") as f:
        json.dump(conf_dict, f, indent=4)


def group_configurations(exp_file, conf_file):
    with open(exp_file) as f:
        experiments = json.load(f)

    # remove experiments with empty files
    c2f = copy.deepcopy(experiments)
    c2f = {file: conf for file, conf in c2f.items() if conf is not None}

    # remove position from experiment configuration
    for conf in c2f.values():
        conf["staNode"].pop("position")

    # convert configuration of each experiment to unique string
    c2f = {file: json.dumps(conf, sort_keys=True, separators=None) for file, conf in c2f.items()}

    # group experiment files for configuration string
    configs = {conf: [] for conf in c2f.values()}
    for file, conf in c2f.items():
        configs[conf].append(file)

    # output number of different configurations
    print(len(configs))

    # extract STA position for each experiment file
    file_to_pos = {file: conf["staNode"]["position"] for file, conf in experiments.items() if conf is not None}

    # list of experiment configurations, including file and STA position for each file
    configs = [{"config": json.loads(conf_str), "files": {f: file_to_pos[f] for f in files}} for conf_str, files in configs.items()]

    with open(conf_file, "w") as f:
        json.dump(configs, f, indent=4)


def create_datasets(exp_dir, dataset_dir, conf_file, features):
    with open(conf_file) as f:
        json_confs = json.load(f)
        for idx, conf in enumerate(json_confs):
            map_dir = dataset_dir / "map_{}".format(idx) / "data"
            map_dir.mkdir(parents=True, exist_ok=True)
            for fname in conf["files"]:
                with open(exp_dir / fname) as f:
                    data = json.load(f)[1:-1]
                    with open(map_dir/fname, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(features.keys())
                        for row in data:
                            writer.writerow(f(row) for f in features.values())


def create_dataset_info(conf_file, dataset_dir):
    with open(conf_file) as f:
        configs = json.load(f)

    for idx, conf in enumerate(configs):
        files = conf["files"]
        x_pos = list(set(map(lambda pos: pos["x"], files.values())))
        y_pos = list(set(map(lambda pos: pos["y"], files.values())))
        
        min_x, max_x = min(x_pos), max(x_pos)
        min_y, max_y = min(y_pos), max(y_pos)

        step_x = (max_x - min_x) / (len(x_pos) - 1)
        step_y = (max_y - min_y) / (len(y_pos) - 1)

        x_size = int((max_x - min_x) / step_x) + 1
        y_size = int((max_y - min_y) / step_y) + 1

        file_map = [[None]*y_size for _ in range(x_size)]

        for fname, pos in files.items():
            file_map[round((pos["x"] - min_x)/step_x)][round((pos["y"] - min_x)/step_x)] = fname

        with open(dataset_dir / "map_{}".format(idx) / "file_map.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            for row in file_map:
                writer.writerow(row)
        
        ap_pos = conf["config"]["apNodes"][0]["position"]
        info_json = {
            "range": ((min_x, max_x), (min_y, max_y)),
            "shape": (x_size, y_size),
            "step": (step_x, step_y),
            "ap_pos": (round(ap_pos["x"]), round(ap_pos["y"]), round(ap_pos["z"]))
        }

        with open(dataset_dir / "map_{}".format(idx) / "info.json", 'w') as f:
            json.dump(info_json, f, indent=4)
        

#%%
def main():
    extract_headers(EXP_DIR, EXP_FILE)  # extract experiment configuration for each file

    group_configurations(EXP_FILE, CONF_FILE)  # group single experiments by APs and interferents setup

    create_datasets(EXP_DIR, DATASET_DIR, CONF_FILE, FEATURES) # create csvs with feature from simulation log

    create_dataset_info(CONF_FILE, DATASET_DIR) # create information files for maps


if __name__ == "__main__":
    main()
