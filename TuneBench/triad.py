"""
Triad benchmark.
"""

# Templates
TEMPLATE_OPENCL = """__kernel void(__global const <%VECTOR_DATA%> * const restrict A, __global const <%VECTOR_DATA%> * const restrict B, __global <%VECTOR_DATA%> * const restrict C) {
unsigned int item = (get_group_id(0) * <%ITEMS_PER_WORKGROUP%>) + get_local_id(0);
<%COMPUTE%>
}\n"""
TEMPLATE_CUDA = """__global__ void(const <%VECTOR_DATA%> * const restrict A, const <%VECTOR_DATA%> * const restrict B, <%VECTOR_DATA%> * const restrict C) {
unsigned int item = (blockIdx.x * <%THREADS_PER_BLOCK%>) + threadIdx.x;
<%COMPUTE%>
}\n"""
TEMPLATE_COMPUTE = """C[item + <%OFFSET%>] = A[item + <%OFFSET%>] + (<%FACTOR%> * B[item + <%OFFSET%>]);"""

# Code generator
def generate_triad_code(language, configuration, factor):
    """
    Generate the source code for triad.
    """
    code = str()
    if language == "opencl":
        code = TEMPLATE_OPENCL.replace("<%VECTOR_DATA%>", configuration["vector_data"])
        code = code.replace("<%ITEMS_PER_WORKGROUP%>", str(int(configuration["threads_dim0"]) * int(configuration["items_dim0"])))
    elif language == "cuda":
        code = TEMPLATE_CUDA.replace("<%VECTOR_DATA%>", configuration["vector_data"])
        code = code.replace("<%THREADS_PER_BLOCK%>", str(int(configuration["threads_dim0"]) * int(configuration["items_dim0"])))
    compute_code = str()
    for item in range(int(configuration["items_dim0"])):
        temp = TEMPLATE_COMPUTE
        if item == 0:
            temp = temp.replace(" + <%OFFSET%>", "")
        else:
            temp = temp.replace("<%OFFSET%>", str(item * int(configuration["threads_dim0"])))
        temp = temp.replace("<%FACTOR%>", str(factor))
        # Format the code
        if item != (configuration["items_dim0"] - 1):
            temp = temp + "\n"
        compute_code = compute_code + temp
    code = code.replace("<%COMPUTE%>", compute_code)

    return code

# Configurations generator
def generate_triad_configurations(parameters):
    """
    Generate all configurations for triad.
    """
    configurations = list()
    for vector in range(parameters["vector_data"]["start"], parameters["vector_data"]["stop"], parameters["vector_data"]["increment"]):
        if vector == 1:
            data_type = parameters["vector_data"]["type"]
        else:
            data_type = parameters["vector_data"]["type"] + str(vector)
        for threads in range(parameters["threads_dim0"]["start"], parameters["threads_dim0"]["stop"], parameters["threads_dim0"]["increment"]):
            for items in range(parameters["items_dim0"]["start"], parameters["items_dim0"]["stop"], parameters["items_dim0"]["increment"]):
                configurations.append(dict(vector_data=data_type, threads_dim0=threads, items_dim0=items))
    return configurations

# Control implementation
def triad_control(A, B, factor):
    """
    Control CPU implementation.
    """
    control = list()
    for item in range(0, len(A)):
        control.append(A[item] + (B[item] * factor))
    return control
