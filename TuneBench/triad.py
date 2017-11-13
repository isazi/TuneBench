"""
Triad benchmark.
"""

import numpy
from kernel_tuner import tune_kernel

# Global variables
FACTOR = 42.0
DATA_TYPE = "float"

# Code templates
TEMPLATE_OPENCL = """__kernel void triad(__global const """ + DATA_TYPE + """<%VECTOR_SIZE%> * const restrict A, __global const """ + DATA_TYPE + """<%VECTOR_SIZE%> * const restrict B, __global """ + DATA_TYPE + """<%VECTOR_SIZE%> * const restrict C) {
unsigned int item = (get_group_id(0) * <%ITEMS_PER_WORKGROUP%>) + get_local_id(0);
<%COMPUTE%>
}\n"""
TEMPLATE_CUDA = """#include <helper_math.h>
extern "C" {
__global__ void triad(const """ + DATA_TYPE + """<%VECTOR_SIZE%> * A, const """ + DATA_TYPE + """<%VECTOR_SIZE%> * B, """ + DATA_TYPE + """<%VECTOR_SIZE%> * C) {
unsigned int item = (blockIdx.x * <%THREADS_PER_BLOCK%>) + threadIdx.x;
<%COMPUTE%>
}
}\n"""
TEMPLATE_COMPUTE = """C[item + <%OFFSET%>] = A[item + <%OFFSET%>] + (<%FACTOR%> * B[item + <%OFFSET%>]);"""

# OpenCL code generator
def generate_code_OpenCL(configuration):
    """
    OpenCL code generator for the Triad benchmark.
    """
    code = str()
    if configuration["vector_size"] == 1:
        code = TEMPLATE_OPENCL.replace("<%VECTOR_SIZE%>", "")
    else:
        code = TEMPLATE_OPENCL.replace("<%VECTOR_SIZE%>", str(configuration["vector_size"]))
    code = code.replace("<%ITEMS_PER_WORKGROUP%>", str(int(configuration["threads_dim0"]) * int(configuration["items_dim0"])))
    compute_code = str()
    for item in range(configuration["items_dim0"]):
        temp = TEMPLATE_COMPUTE
        if item == 0:
            temp = temp.replace(" + <%OFFSET%>", "")
        else:
            temp = temp.replace("<%OFFSET%>", str(item * configuration["threads_dim0"]))
        temp = temp.replace("<%FACTOR%>", str(FACTOR) + "f")
        # Format the code
        if item != (configuration["items_dim0"] - 1):
            temp = temp + "\n"
        compute_code = compute_code + temp
    code = code.replace("<%COMPUTE%>", compute_code)

    return code

# CUDA code generator
def generate_code_CUDA(configuration):
    """
    CUDA code generator for the Triad benchmark.
    """
    code = str()
    if configuration["vector_size"] == 1:
        code = TEMPLATE_CUDA.replace("<%VECTOR_SIZE%>", "")
    else:
        code = TEMPLATE_CUDA.replace("<%VECTOR_SIZE%>", str(configuration["vector_size"]))
    code = code.replace("<%THREADS_PER_BLOCK%>", str(int(configuration["threads_dim0"]) * int(configuration["items_dim0"])))
    compute_code = str()
    for item in range(configuration["items_dim0"]):
        temp = TEMPLATE_COMPUTE
        if item == 0:
            temp = temp.replace(" + <%OFFSET%>", "")
        else:
            temp = temp.replace("<%OFFSET%>", str(item * configuration["threads_dim0"]))
        if configuration["vector_size"] == 1:
            temp = temp.replace("<%FACTOR%>", str(FACTOR) + "f")
        else:
            factor = "make_" + DATA_TYPE + str(configuration["vector_size"]) + "("
            for vector in range(0, configuration["vector_size"]):
                factor = factor + str(FACTOR) + "f"
                if vector < configuration["vector_size"] - 1:
                    factor = factor + ", "
            factor = factor + ")"
            temp = temp.replace("<%FACTOR%>", factor)
        # Format the code
        if item != (configuration["items_dim0"] - 1):
            temp = temp + "\n"
        compute_code = compute_code + temp
    code = code.replace("<%COMPUTE%>", compute_code)

    return code

def tune(input_size, language, constraints):
    """
    Triad benchmark tuner.
    """

    A = numpy.random.randn(input_size).astype(numpy.float32)
    B = numpy.random.randn(input_size).astype(numpy.float32)
    C = numpy.random.randn(input_size).astype(numpy.float32)
    kernel_arguments = [A, B, C]
    control = [None, None, A + (FACTOR * B)]

    tuning_parameters = dict()
    tuning_parameters["threads_dim0"] = [threads for threads in range(constraints["threads_dim0_min"], constraints["threads_dim0_max"] + 1, constraints["threads_dim0_step"])]
    tuning_parameters["items_dim0"] = [items for items in range(constraints["items_dim0_min"], constraints["items_dim0_max"] + 1, constraints["items_dim0_step"])]
    dim0_divisor = ["threads_dim0 * items_dim0 * vector_size"]
    restrictions = ["(" + str(input_size) + " % (threads_dim0 * items_dim0 * vector_size)) == 0", "(items_dim0 * vector_size) <= " + str(constraints["items_dim0_max"]) ]
    
    if language == "OpenCL":
        tuning_parameters["vector_size"] = [2**i for i in range(5)]
        results = tune_kernel("triad", generate_code_OpenCL, input_size, kernel_arguments, tuning_parameters, grid_div_x=dim0_divisor, restrictions=restrictions, lang=language, answer=control)
    else:
        tuning_parameters["vector_size"] = [2**i for i in range(3)]
        results = tune_kernel("triad", generate_code_CUDA, input_size, kernel_arguments, tuning_parameters, grid_div_x=dim0_divisor, restrictions=restrictions, lang=language, answer=control)
