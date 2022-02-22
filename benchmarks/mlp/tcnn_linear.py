import time
import json
import string
import argparse

import torch
import torch.nn as nn

import tinycudann as tcnn


def main(args):
    print(f"Tiny Cuda NN")
    for n_features in args.n_features:
        for n_layers in args.n_layers:
            bench(args.device_id, n_features, n_layers, args.forward_only)


def bench(device_id, n_features, n_layers, forward_only):
    B, R, N, D = 4, 512, 256, 3
    device = torch.device(f"cuda:{device_id}")
    n_iters = 100

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
        "output_activation": "None",
        "n_neurons": ${n_features},
        "n_hidden_layers": ${n_layers}
    }
    }'''

    json_template = string.Template(json_str)
    config = json.loads(json_template.safe_substitute({"n_features": n_features,
                                                       "n_layers": n_layers}))
    
    ## with open("./model_sample.json") as fp:
    ##     config = json.load(fp)
                 
    ## model = tcnn.NetworkWithInputEncoding(
    ##     3, n_features,
    ##     config["encoding"], config["network"]
    ## )

    ## encoding = tcnn.Encoding(n_input_dims, config["encoding"])
    model = tcnn.Network(3, n_features, config["network"])

    x = torch.randn(B*R*N, D, dtype=torch.float32, requires_grad=True).to(device)
    
    def evaluate():
        for i in range(n_iters):
            y = model(x)
            y = torch.sum(y)
            if not forward_only:
                y.backward()
        y.cpu()
        
    evaluate()
    st = time.perf_counter()
    evaluate()
    et = time.perf_counter() - st
    print(f"{n_features},{n_layers},{et}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=str, default="0")
    parser.add_argument("--n_layers", type=int, nargs="+", default=[12, 24, 36])
    parser.add_argument("--n_features", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--forward_only", action="store_true")
    
    args = parser.parse_args()
    
    main(args)

