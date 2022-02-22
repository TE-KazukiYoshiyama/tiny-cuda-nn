import time
import argparse

import torch
import torch.nn as nn
import torch.jit as jit


class ActLinear(nn.Module):

    def __init__(self, n_features0, n_features1):
        super(ActLinear, self).__init__()

        self.linear = nn.Linear(n_features0, n_features1, bias=False)
        self.act = nn.ReLU()

        
    def forward(self, x):
        h = self.linear(x)
        h = self.act(h)
        return h
        

class MLP(nn.Module):

    def __init__(self, n_features=64, n_layers=8):
        super(MLP, self).__init__()

        self.stem_linear = ActLinear(3, n_features)
        self.stem_act = nn.ReLU()
        
        linears = []
        for i in range(n_layers):
            linear = ActLinear(n_features, n_features)
            linears.append(linear)
        self.linears = nn.Sequential(*linears)

        self.out_linear = nn.Linear(n_features, n_features)

        
    def forward(self, x):
        h = self.stem_linear(x)
        h = self.linears(h)
        h = self.out_linear(h)
        return h

    
def main(args):
    print(f"PyTorch({torch.__version__})")
    for type_config in args.type_configs:
        for n_features in args.n_features:
            for n_layers in args.n_layers:
                bench(args.device_id, type_config, n_features, n_layers, args.forward_only)

                
def bench(device_id, type_config, n_features, n_layers, forward_only):
    B, R, N, D = 4, 512, 256, 3
    device = torch.device(f"cuda:{device_id}")
    n_iters = 100

    mlp = MLP(n_features, n_layers).to(device)
    x = torch.randn(B, R, N, D, dtype=torch.float32, requires_grad=True).to(device)
#    mlp_traced = torch.jit.trace(mlp, (x, ))
    
    def evaluate():
        if type_config == "half":
            for i in range(n_iters):
                with torch.cuda.amp.autocast():
                    y = mlp(x)
                    y = torch.sum(y)
                if not forward_only:
                    y.backward()
        elif type_config == "float":
            for i in range(n_iters):
                y = mlp(x)
                y = torch.sum(y)
                if not forward_only:
                    y.backward()
        y.cpu()

    evaluate()
    st = time.perf_counter()
    evaluate()
    et = time.perf_counter() - st
    print(f"{type_config},{n_features},{n_layers},{et}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=str, default="0")
    parser.add_argument("--type_configs", type=str, nargs="+", default=["float", "half"])
    parser.add_argument("--n_layers", type=str, nargs="+", default=[12, 24, 36])
    parser.add_argument("--n_features", type=str, nargs="+", default=[32, 64, 128])
    parser.add_argument("--forward_only", action="store_true")
    
    args = parser.parse_args()
    
    main(args)
