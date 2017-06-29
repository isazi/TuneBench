// Copyright 2016 Alessio Sclocco <a.sclocco@vu.nl>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include <iostream>
#include <exception>
#include <iomanip>
#include <random>
#include <algorithm>

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <Stencil.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>


void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< inputDataType > * input, cl::Buffer * input_d, cl::Buffer * output_d, const unsigned int outputSize);

int main(int argc, char * argv[]) {
  bool reInit = true;
  unsigned int nrIterations = 0;
  unsigned int samplingFactor = 100;
  unsigned int clPlatformID = 0;
  unsigned int clDeviceID = 0;
  unsigned int vectorSize = 0;
  unsigned int maxThreads = 0;
  unsigned int maxItems = 0;
  unsigned int matrixWidth = 0;
  unsigned int padding = 0;
  // Random number generation
  std::random_device randomDevice;
  std::default_random_engine randomEngine(randomDevice());
  std::uniform_int_distribution<inputDataType> uniformDistribution(0, magicValue);

  try {
    isa::utils::ArgumentList args(argc, argv);

    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    if ( args.getSwitch("-sampling") ) {
      samplingFactor = args.getSwitchArgument< unsigned int >("-factor");
    }
    nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
    vectorSize = args.getSwitchArgument< unsigned int >("-vector");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    matrixWidth = args.getSwitchArgument< unsigned int >("-matrix_width");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -opencl_platform ... -opencl_device ... [-sampling -factor ...] -iterations ... -vector ... -padding ... -max_threads ... -max_items ... -matrix_width ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  cl::Context clContext;
  std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
  std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
  std::vector< std::vector< cl::CommandQueue > > * clQueues = 0;

  // Allocate host memory
  std::vector<inputDataType> input((matrixWidth + 2) * isa::utils::pad(matrixWidth + 2, padding)), output(matrixWidth * isa::utils::pad(matrixWidth, padding)), output_c;
  cl::Buffer input_d, output_d;

  for ( unsigned int y = 0; y < matrixWidth + 2; y++ ) {
    for ( unsigned int x = 0; x < matrixWidth + 2; x++ ) {
      if ( y == 0 || y == (matrixWidth - 1) ) {
        input[(y * isa::utils::pad(matrixWidth + 2, padding)) + x] = 0;
      } else if ( x == 0 || x == (matrixWidth - 1) ) {
        input[(y * isa::utils::pad(matrixWidth + 2, padding)) + x] = 0;
      } else {
        input[(y * isa::utils::pad(matrixWidth + 2, padding)) + x] = uniformDistribution(randomEngine);
      }
    }
  }
  output_c.resize(output.size());

  // Run the control
  TuneBench::stencil2D(input, output_c, matrixWidth, padding);

  // Generate tuning configurations
  std::vector<TuneBench::Stencil2DConf> configurations;
  for ( unsigned int threadsD0 = vectorSize; threadsD0 <= maxThreads; threadsD0 += vectorSize) {
    for ( unsigned int threadsD1 = 1; threadsD0 * threadsD1 <= maxThreads; threadsD1++ ) {
      for ( unsigned int itemsD0 = 1; itemsD0 <= maxItems; itemsD0++ ) {
        if ( matrixWidth % (threadsD0 * itemsD0) != 0 ) {
          continue;
        }
        for ( unsigned int itemsD1 = 1; itemsD0 * itemsD1 <= maxItems; itemsD1++ ) {
          if ( matrixWidth % (threadsD1 * itemsD1) != 0 ) {
            continue;
          }
          for ( unsigned int local = 0; local < 2; local++ ) {
            TuneBench::Stencil2DConf configuration;
            configuration.setLocalMemory(static_cast<bool>(local));
            configuration.setNrThreadsD0(threadsD0);
            configuration.setNrThreadsD1(threadsD1);
            configuration.setNrItemsD0(itemsD0);
            configuration.setNrItemsD1(itemsD1);
            configurations.push_back(configuration);
          }
        }
      }
    }
  }
  if ( samplingFactor < 100 ) {
    unsigned int newSize = static_cast<unsigned int>((configurations.size() * samplingFactor) / 100.0);
    std::shuffle(configurations.begin(), configurations.end(), randomEngine);
    configurations.resize(newSize);
  }

  std::cout << std::fixed << std::endl;
  std::cout << "# matrixWidth *configuration* GFLOP/s time stdDeviation COV" << std::endl << std::endl;

  for ( auto configuration = configurations.begin(); configuration != configurations.end(); ++configuration ) {
    // Generate kernel
    double gflops = isa::utils::giga(static_cast< uint64_t >(matrixWidth) * matrixWidth * 18.0);
    cl::Event clEvent;
    cl::Kernel * kernel;
    isa::utils::Timer timer;
    std::string * code = TuneBench::getStencil2DOpenCL(*configuration, inputDataName, matrixWidth, padding);

    if ( reInit ) {
      delete clQueues;
      clQueues = new std::vector< std::vector< cl::CommandQueue > >();
      isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);
      try {
        initializeDeviceMemory(clContext, &(clQueues->at(clDeviceID)[0]), &input, &input_d, &output_d, output.size());
      } catch ( cl::Error & err ) {
        return -1;
      }
      reInit = false;
    }
    try {
      kernel = isa::OpenCL::compile("stencil2D", *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      delete code;
      continue;
    }
    delete code;

    cl::NDRange global(matrixWidth / (*configuration).getNrItemsD0(), matrixWidth / (*configuration).getNrItemsD1());
    cl::NDRange local((*configuration).getNrThreadsD0(), (*configuration).getNrThreadsD1());

    kernel->setArg(0, input_d);
    kernel->setArg(1, output_d);

    try {
      // Warm-up run
      clQueues->at(clDeviceID)[0].finish();
      clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &clEvent);
      clEvent.wait();
      // Tuning runs
      for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
        timer.start();
        clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &clEvent);
        clEvent.wait();
        timer.stop();
      }
      clQueues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(inputDataType), reinterpret_cast< void * >(output.data()), 0, &clEvent);
      clEvent.wait();
    } catch ( cl::Error & err ) {
      reInit = true;
      std::cerr << "OpenCL kernel execution error (";
      std::cerr << (*configuration).print();
      std::cerr << "): ";
      std::cerr << isa::utils::toString(err.err()) << std::endl;
      delete kernel;
      if ( err.err() == -4 || err.err() == -61 ) {
        return -1;
      }
      continue;
    }
    delete kernel;

    bool error = false;
    for ( unsigned int y = 0; y < matrixWidth; y++ ) {
      for ( unsigned int x = 0; x < matrixWidth; x++ ) {
        if ( !isa::utils::same(output[(y * isa::utils::pad(matrixWidth, padding)) + x], output_c[(y * isa::utils::pad(matrixWidth, padding)) + x]) ) {
          std::cerr << "Output error (" << (*configuration).print() << ")." << std::endl;
          error = true;
          break;
        }
      }
      if ( error ) {
        break;
      }
    }
    if ( error ) {
      continue;
    }

    std::cout << matrixWidth << " ";
    std::cout << (*configuration).print() << " ";
    std::cout << std::setprecision(3);
    std::cout << gflops / timer.getAverageTime() << " ";
    std::cout << std::setprecision(6);
    std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation() << std::endl;
  }
  std::cout << std::endl;

  return 0;
}

void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< inputDataType > * input, cl::Buffer * input_d, cl::Buffer * output_d, const unsigned int outputSize) {
  try {
    *input_d = cl::Buffer(clContext, CL_MEM_READ_ONLY, input->size() * sizeof(inputDataType), 0, 0);
    *output_d = cl::Buffer(clContext, CL_MEM_WRITE_ONLY, outputSize * sizeof(inputDataType), 0, 0);
    clQueue->enqueueWriteBuffer(*input_d, CL_FALSE, 0, input->size() * sizeof(inputDataType), reinterpret_cast< void * >(input->data()));
    clQueue->finish();
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error (memory initialization): " << isa::utils::toString(err.err()) << "." << std::endl;
    throw;
  }
}

