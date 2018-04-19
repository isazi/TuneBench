"""
Triad benchmark.
"""

import math
import numpy
from kernel_tuner import tune_kernel

class Triad:
    language = str()
    type = str()
    input_size = int()
    factor = float()
    
    TEMPLATE_OPENCL = """__kernel void triad(__global const <%TYPE%><%VECTOR_SIZE%> * const restrict A, __global const <%TYPE%><%VECTOR_SIZE%> * const restrict B, __global <%TYPE%><%VECTOR_SIZE%> * const restrict C) {
        unsigned int item = (get_group_id(0) * <%ITEMS_PER_WORKGROUP%>) + get_local_id(0);
        <%COMPUTE%>
    }\n"""
    TEMPLATE_CUDA = """#include <helper_math.h>
        extern "C" {
        __global__ void triad(const <%TYPE%><%VECTOR_SIZE%> * A, const <%TYPE%><%VECTOR_SIZE%> * B, <%TYPE%><%VECTOR_SIZE%> * C) {
        unsigned int item = (blockIdx.x * <%THREADS_PER_BLOCK%>) + threadIdx.x;
        <%COMPUTE%>
        }
    }\n"""
    TEMPLATE_COMPUTE = """C[item + <%OFFSET%>] = A[item + <%OFFSET%>] + (<%FACTOR%> * B[item + <%OFFSET%>]);"""

    def __init__(self, language, type, input_size, factor):
        self.language = language
        self.type = type
        self.input_size = input_size
        self.factor = factor

    # OpenCL code generator
    def generate_code_OpenCL(self, configuration):
        code = self.TEMPLATE_OPENCL.replace("<%TYPE%>", self.type)
        if configuration["vector_size"] == 1:
            code = code.replace("<%VECTOR_SIZE%>", "")
        else:
            code = code.replace("<%VECTOR_SIZE%>", str(configuration["vector_size"]))
        code = code.replace("<%ITEMS_PER_WORKGROUP%>", str(int(configuration["threads_dim0"]) * int(configuration["items_dim0"])))
        compute_code = str()
        for item in range(configuration["items_dim0"]):
            temp = self.TEMPLATE_COMPUTE
            if item == 0:
                temp = temp.replace(" + <%OFFSET%>", "")
            else:
                temp = temp.replace("<%OFFSET%>", str(item * configuration["threads_dim0"]))
            temp = temp.replace("<%FACTOR%>", str(self.factor) + "f")
            # Format the code
            if item != (configuration["items_dim0"] - 1):
                temp = temp + "\n"
            compute_code = compute_code + temp
        code = code.replace("<%COMPUTE%>", compute_code)
        return code

    # CUDA code generator
    def generate_code_CUDA(self, configuration):
        code = self.TEMPLATE_CUDA.replace("<%TYPE%>", self.type)
        if configuration["vector_size"] == 1:
            code = code.replace("<%VECTOR_SIZE%>", "")
        else:
            code = code.replace("<%VECTOR_SIZE%>", str(configuration["vector_size"]))
        code = code.replace("<%THREADS_PER_BLOCK%>", str(int(configuration["threads_dim0"]) * int(configuration["items_dim0"])))
        compute_code = str()
        for item in range(configuration["items_dim0"]):
            temp = self.TEMPLATE_COMPUTE
            if item == 0:
                temp = temp.replace(" + <%OFFSET%>", "")
            else:
                temp = temp.replace("<%OFFSET%>", str(item * configuration["threads_dim0"]))
            if configuration["vector_size"] == 1:
                temp = temp.replace("<%FACTOR%>", str(self.factor) + "f")
            else:
                factor = "make_" + self.type + str(configuration["vector_size"]) + "("
                for vector in range(0, configuration["vector_size"]):
                    factor = factor + str(self.factor) + "f"
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

    @staticmethod
    def verify(control_data, data, atol=None):
        result = numpy.allclose(control_data, data, atol)
        if result is False:
            numpy.set_printoptions(precision=6, suppress=True)
            print(control_data)
            print(data)
        return result

    def tune(self, constraints, numpy_type):
        B = numpy.random.randn(self.input_size).astype(numpy_type)
        C = numpy.random.randn(self.input_size).astype(numpy_type)
        A = numpy.random.randn(self.input_size).astype(numpy_type)
        kernel_arguments = [A, B, C]
        control = [None, None, A + (self.factor * B)]

        tuning_parameters = dict()
        tuning_parameters["threads_dim0"] = [threads for threads in range(constraints["threads_dim0_min"], constraints["threads_dim0_max"] + 1, constraints["threads_dim0_step"])]
        tuning_parameters["threads_dim1"] = [1]
        tuning_parameters["threads_dim2"] = [1]
        tuning_parameters["items_dim0"] = [items for items in range(constraints["items_dim0_min"], constraints["items_dim0_max"] + 1, constraints["items_dim0_step"])]
        dim0_divisor = ["threads_dim0 * items_dim0 * vector_size"]
        restrictions = ["(" + str(self.input_size) + " % (threads_dim0 * items_dim0 * vector_size)) == 0", "(items_dim0 * vector_size) <= " + str(constraints["items_dim0_max"]) ]
        block_size_names=["threads_dim0", "threads_dim1", "threads_dim2"]
        
        try:
            if self.language == "OpenCL":
                tuning_parameters["vector_size"] = [2**i for i in range(5)]
                results = tune_kernel("triad", self.generate_code_OpenCL, self.input_size, kernel_arguments, tuning_parameters, grid_div_x=dim0_divisor, block_size_names=block_size_names, restrictions=restrictions, lang=self.language, answer=control, verify=self.verify)
            else:
                tuning_parameters["vector_size"] = [2**i for i in range(3)]
                results = tune_kernel("triad", self.generate_code_CUDA, self.input_size, kernel_arguments, tuning_parameters, grid_div_x=dim0_divisor, block_size_names=block_size_names, restrictions=restrictions, lang=self.language, answer=control, verify=self.verify)
        except Exception as error:
            print(error)
        return results