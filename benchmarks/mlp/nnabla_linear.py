import time
import argparse

import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.ext_utils import get_extension_context


def mlp(x, n_features, n_layers=8):
    base_axis = x.ndim - 1
    h = PF.affine(x, n_features, base_axis=base_axis, with_bias=False, name="stem-linear")
    h = F.relu(h)
    for l in range(n_layers):
        h = PF.affine(h, n_features, base_axis=base_axis, with_bias=False, name=f"linear-{l:02d}")
        h = F.relu(h)
    h = PF.affine(h, n_features, base_axis=base_axis, with_bias=False, name=f"out-linear")
    return h


def main(args):
    print(f"NNabla({nn.__version__})")
    for type_config in args.type_configs:
        for n_features in args.n_features:
            for n_layers in args.n_layers:
                nn.clear_parameters()
                bench(args.device_id, type_config, n_features, n_layers, args.forward_only)


def bench(device_id, type_config, n_features, n_layers, forward_only):
    B, R, N, D = 4, 512, 256, 3
    n_iters = 100

    ctx = get_extension_context("cudnn", device_id=device_id, type_config=type_config)
    nn.set_default_context(ctx)
    
    x_data = np.random.randn(B, R, N, D)
    x = nn.Variable.from_numpy_array(x_data).apply(need_grad=True)
    y = mlp(x, n_features, n_layers)
    y = F.sum(y)
    
    def evaluate():
        for i in range(n_iters):
            y.forward(clear_no_need_grad=True)
            if not forward_only:
                y.backward(clear_buffer=True)
        y.d    
    
    evaluate()

    st = time.perf_counter()
    evaluate()
    et = time.perf_counter() - st
    print(f"{type_config},{n_features},{n_layers},{et}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=str, default="0")
    parser.add_argument("--type_configs", type=list, nargs="+", default=["float", "half"])
    parser.add_argument("--n_layers", type=list, nargs="+", default=[12, 24, 36])
    parser.add_argument("--n_features", type=list, nargs="+", default=[32, 64, 128])
    parser.add_argument("--forward_only", action="store_true")

    args = parser.parse_args()
    
    main(args)
