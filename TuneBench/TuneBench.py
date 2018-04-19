
import argparse
import numpy

# Benchmarks
import Triad

def parseCommandLine():
    parser = argparse.ArgumentParser(description="TuneBench: a fully tunable benchmark for many-core accelerators")
    parser.add_argument("--language", help="Language: CUDA or OpenCL", choices=["CUDA", "OpenCL"], required=True)
    parser_benchmarks = parser.add_subparsers(dest="benchmark")
    parser_triad = parser_benchmarks.add_parser("triad")
    parser_triad.add_argument("--type", help="Data type on the device.", choices=["float", "double"], type=str, required=True)
    parser_triad.add_argument("--input_size", help="Input size", type=int, required=True)
    parser_triad.add_argument("--factor", help="Scalar factor.", type=float, required=True)
    parser_triad.add_argument("--threads_dim0_min", help="Minimum number of threads in dimension 0.", type=int, required=True)
    parser_triad.add_argument("--threads_dim0_max", help="Maximum number of threads in dimension 0.", type=int, required=True)
    parser_triad.add_argument("--threads_dim0_step", help="Thread increment in dimension 0.", type=int, required=True)
    parser_triad.add_argument("--items_dim0_min", help="Minimum number of variables for dimension 0.", type=int, required=True)
    parser_triad.add_argument("--items_dim0_max", help="Maximum number of variables for dimension 0.", type=int, required=True)
    parser_triad.add_argument("--items_dim0_step", help="Variable increment for dimension 0.", type=int, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    arguments = parseCommandLine()
    if arguments.benchmark == "triad":
        constraints = dict()
        constraints["threads_dim0_min"] = arguments.threads_dim0_min
        constraints["threads_dim0_max"] = arguments.threads_dim0_max
        constraints["threads_dim0_step"] = arguments.threads_dim0_step
        constraints["items_dim0_min"] = arguments.items_dim0_min
        constraints["items_dim0_max"] = arguments.items_dim0_max
        constraints["items_dim0_step"] = arguments.items_dim0_step
        numpy_type = None
        if arguments.type == "float":
            numpy_type = numpy.float32
        elif arguments.type == "double":
            numpy_type = numpy.float64
        kernel = Triad.Triad(arguments.language, arguments.type, arguments.input_size, arguments.factor)
        results = kernel.tune(constraints, numpy_type)
