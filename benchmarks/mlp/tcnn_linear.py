import time
import json
import string
import argparse

import torch
import torch.nn as nn

import tinycudann as tcnn


def bench(device_id, n_iters, W, L, p, forward_only, inference_only, warmup):
    B, D = 2 ** p, W
    device = torch.device(f"cuda:{device_id}")

    json_str = '''{
    "loss": {
        "otype": "L2"
    },
    "optimizer": {
        "otype": "Adam",
        "learning_rate": 1e-3
    },
    "encoding": {
        "otype": "Identity"
    },
    "network": {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "ReLU",
        "n_neurons": ${W},
        "n_hidden_layers": ${L}
    }
    }'''

    json_template = string.Template(json_str)
    config = json.loads(json_template.safe_substitute({"W": W,
                                                       "L": L}))
    
    ## with open("./model_sample.json") as fp:
    ##     config = json.load(fp)
                 
    ## model = tcnn.NetworkWithInputEncoding(
    ##     3, n_features,
    ##     config["encoding"], config["network"]
    ## )

    ## encoding = tcnn.Encoding(n_input_dims, config["encoding"])

    def watch_time(func):
        def wrapper(*args, **kwargs):
            st = time.perf_counter()
            func(*args, **kwargs)
            et = time.perf_counter()
            return et - st
        return wrapper

    D_input = W
    model = tcnn.Network(D_input, W, config["network"])

    # the grad computation wrt the first input is very slow..., so set as False
    x = torch.randn(B, D_input, dtype=torch.float16, requires_grad=False).to(device)
    dy = torch.randn(B, W, dtype=torch.float16).to(device)

    @watch_time
    def bench_ffmlp():
        for i in range(n_iters):
            if inference_only:
                with torch.no_grad():
                    y = model(x)
            else:
                y = model(x)
            if not (forward_only or inference_only):
                y.backward(dy)
        y.cpu()
        
    et_ffmlp = bench_ffmlp()
    if not warmup:
        print(f"2**{p}, {W}, {L}, {et_ffmlp}")

    
def main(args):
    for L in args.n_layers:
        for W in args.widths:
            for p in args.powers:
                bench(args.device_id, args.n_iters, W, L, p, args.forward_only, args.inference_only, True)
                bench(args.device_id, args.n_iters, W, L, p, args.forward_only, args.inference_only, False)

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=str, default="0")
    parser.add_argument('--n_iters', type=int, default=100, help="")
    parser.add_argument('--powers', type=int, nargs='+', help="2**this is batch size")
    parser.add_argument("--n_layers", type=int, nargs="+")
    parser.add_argument('--widths', type=int, nargs='+', help="")
    parser.add_argument("--forward_only", action="store_true")
    parser.add_argument("--inference_only", action="store_true")
    
    args = parser.parse_args()
    
    main(args)

