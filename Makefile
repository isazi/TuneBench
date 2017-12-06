
# https://github.com/isazi/utils
UTILS := $(HOME)/src/utils
# https://github.com/isazi/OpenCL
OPENCL := $(HOME)/src/OpenCL

INCLUDES := -I"include" -I"$(UTILS)/include"
CL_INCLUDES := $(INCLUDES) -I"$(OPENCL)/include"
CL_LIBS := -L"$(OPENCL_LIB)" -L"$(UTILS)/lib" -L"$(OPENCL)/lib"

CFLAGS := -std=c++11 -Wall
ifneq ($(debug), 1)
	CFLAGS += -O3 -g0 -fopenmp
else
	CFLAGS += -O0 -g3
endif

LDFLAGS := -lm -lutils
CL_LDFLAGS := $(LDFLAGS) -lOpenCL -lisaOpenCL

CC := g++

# Dependencies

all: bin/Reduction.o bin/ReductionTuner bin/ReductionPrint bin/Stencil.o bin/StencilTuner bin/StencilPrint bin/MD.o bin/MDTuner bin/MDPrint bin/Triad.o bin/TriadTuner bin/TriadPrint bin/Correlator.o bin/CorrelatorPrint bin/CorrelatorTuner

bin/Reduction.o: include/Reduction.hpp src/Reduction.cpp
	-@mkdir -p bin
	$(CC) -o bin/Reduction.o -c src/Reduction.cpp $(CL_INCLUDES) $(CFLAGS)

bin/ReductionTuner: bin/Reduction.o include/configuration.hpp src/ReductionTuner.cpp
	-@mkdir -p bin
	$(CC) -o bin/ReductionTuner src/ReductionTuner.cpp bin/Reduction.o $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/ReductionPrint: bin/Reduction.o include/configuration.hpp src/ReductionPrint.cpp
	-@mkdir -p bin
	$(CC) -o bin/ReductionPrint src/ReductionPrint.cpp bin/Reduction.o $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/Stencil.o: include/Stencil.hpp src/Stencil.cpp
	-@mkdir -p bin
	$(CC) -o bin/Stencil.o -c src/Stencil.cpp $(CL_INCLUDES) $(CFLAGS)

bin/StencilTuner: bin/Stencil.o include/configuration.hpp src/StencilTuner.cpp
	-@mkdir -p bin
	$(CC) -o bin/StencilTuner src/StencilTuner.cpp bin/Stencil.o $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/StencilPrint: bin/Stencil.o include/configuration.hpp src/StencilPrint.cpp
	-@mkdir -p bin
	$(CC) -o bin/StencilPrint src/StencilPrint.cpp bin/Stencil.o $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/MD.o: include/MD.hpp src/MD.cpp
	-@mkdir -p bin
	$(CC) -o bin/MD.o -c src/MD.cpp $(CL_INCLUDES) $(CFLAGS)

bin/MDTuner: bin/MD.o include/configuration.hpp src/MDTuner.cpp
	-@mkdir -p bin
	$(CC) -o bin/MDTuner src/MDTuner.cpp bin/MD.o $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/MDPrint: bin/MD.o include/configuration.hpp src/MDPrint.cpp
	-@mkdir -p bin
	$(CC) -o bin/MDPrint src/MDPrint.cpp bin/MD.o $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/Triad.o: include/Triad.hpp src/Triad.cpp
	-@mkdir -p bin
	$(CC) -o bin/Triad.o -c src/Triad.cpp $(CL_INCLUDES) $(CFLAGS)

bin/TriadTuner: include/configuration.hpp bin/Triad.o src/TriadTuner.cpp
	-@mkdir -p bin
	$(CC) -o bin/TriadTuner src/TriadTuner.cpp bin/Triad.o $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/TriadPrint: include/configuration.hpp bin/Triad.o src/TriadPrint.cpp
	-@mkdir -p bin
	$(CC) -o bin/TriadPrint src/TriadPrint.cpp bin/Triad.o $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/Correlator.o: include/Correlator.hpp src/Correlator.cpp
	-@mkdir -p bin
	$(CC) -o bin/Correlator.o -c src/Correlator.cpp $(CL_INCLUDES) $(CFLAGS)

bin/CorrelatorPrint: bin/Correlator.o include/configuration.hpp src/CorrelatorPrint.cpp
	-@mkdir -p bin
	$(CC) -o bin/CorrelatorPrint src/CorrelatorPrint.cpp bin/Correlator.o $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/CorrelatorTuner: bin/Correlator.o include/configuration.hpp src/CorrelatorTuner.cpp
	-@mkdir -p bin
	$(CC) -o bin/CorrelatorTuner src/CorrelatorTuner.cpp bin/Correlator.o $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*

