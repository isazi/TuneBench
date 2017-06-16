"""
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

# Generator
def generate_triad_code(language, configuration, factor):
    code = str()
    if ( language == "opencl" ):
        code = TEMPLATE_OPENCL.replace("<%VECTOR_DATA%>", configuration["vector_data"])
        code = code.replace("<%ITEMS_PER_WORKGROUP%>", str(int(configuration["threads_dim0"]) * int(configuration["items_dim0"])))
    elif ( language == "cuda" ):
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
