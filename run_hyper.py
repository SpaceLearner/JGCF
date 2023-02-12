import argparse
import numpy as np
import yaml
import pynvml
from recbole.quick_start import run_recbole

def get_best_gpu(): # return gpu(torch.device) with largest free memory.
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    deviceMemory = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    return best_device_index # rch.device("cuda:%d"%(best_device_index))

with open("config_amazon_books.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser()
parser.add_argument("--a", default=1.0, help="tuning for a.")
parser.add_argument("--b", default=1.0, help="tuning for a.")
parser.add_argument("--n_layers", default=3, help="tuning for a.")
parser.add_argument("--alpha", default=3.0, help="tuning for alpha.")
parser.add_argument("--beta", default=0.1, help="tuning for beta.")

args = parser.parse_args()
config["a"] = args.a
config["b"] = args.b
config["alpha"] = args.alpha
config['beta'] = args.beta
config['n_layers'] = args.n_layers
config["gpu_id"] = get_best_gpu()

run_recbole(model='LightGCN_Mid_Poly', dataset='Amazon_Books', config_dict=config)