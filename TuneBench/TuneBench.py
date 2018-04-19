
import argparse

# Benchmarks
import Triad

def parseCommandLine():
    parser = argparse.ArgumentParser(description="TuneBench: a fully tunable benchmark for many-core accelerators")
    parser.add_argument("--language", help="Language: CUDA or OpenCL", choices=["CUDA", "OpenCL"], required=True)
    parser_benchmarks = parser.add_subparsers(dest="benchmark")
    parser_triad = parser_benchmarks.add_subparsers("triad")
    parser_triad.add_argument("--type", help="Data type on the device.", type=str)
    parser_triad.add_argument("--numpy_type", help="Data type on the host.", type=str)
    parser_triad.add_argument("--input_size", help="Input size", type=int)
    parser_triad.add_argument("--factor", "Scalar factor.", type=float)
    return parser.parse_args()

if __name__ == "__main__":
    arguments = parseCommandLine()
    if arguments.benchmark == "triad":
        kernel = Triad.Triad(arguments.language, arguments.type, arguments.input_size, arguments.factor)
        results = kernel.tune([], arguments.numpy_type)
