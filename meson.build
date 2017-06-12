project('TuneBench', 'cpp')

# Dependencies
inc = include_directories('include')
opencl = dependency('OpenCL', required : true)
utils_proj = subproject('utils')
opencl_proj = subproject('opencl')

# Executables
executable('TriadTuner', 'src/TriadTuner.cpp', 'src/Triad.cpp', include_directories : [inc, utils_proj.get_variable('inc'), opencl_proj.get_variable('inc')], link_with : [utils_proj.get_variable('lib'), opencl_proj.get_variable('lib')], dependencies : [opencl])
