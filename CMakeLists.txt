# cmake_minimum_required(VERSION 3.11)

# project(RepH LANGUAGES CXX C)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED YES)

# include(ExternalProject)

# find_package(MLIR REQUIRED CONFIG)

# message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
# message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
# option(LLVM_BUILD_TOOLS
#        "Build the LLVM tools. If OFF, just generate build targets." ON)

# set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
# set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
# set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# set(LIT_ARGS_DEFAULT "-sv --timeout=30")
# if(MSVC_IDE OR XCODE)
#   set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
# endif()
# set(LLVM_LIT_ARGS
#     "${LIT_ARGS_DEFAULT}"
#     CACHE STRING "Default options for lit")
# set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" FORCE)

# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
# list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# include(TableGen)
# include(AddLLVM)
# include(AddMLIR)
# include(HandleLLVMOptions)

# # setup python
# include(MLIRDetectPythonEnv)
# mlir_configure_python_dev_packages()

# include_directories(${LLVM_INCLUDE_DIRS})
# include_directories(${MLIR_INCLUDE_DIRS})

# include_directories(${PROJECT_SOURCE_DIR}/mlir/include)
# include_directories(${PROJECT_BINARY_DIR}/mlir/include)

# add_definitions(${LLVM_DEFINITIONS})

# add_subdirectory(mlir)
# # add_subdirectory(tools)



#===================================================================================
#===================================================================================
#===================================================================================



# cmake_minimum_required(VERSION 3.11)

project(RepresentHW LANGUAGES CXX C)

# -----------------------------------------------------------------------------
# Mode detection: standalone vs integrated (superproject)
# - Standalone: RepresentHW is the top-level project (or built separately).
# - Integrated: RepresentHW is added via add_subdirectory()/FetchContent into a
#   superproject that already configures LLVM/MLIR.
# -----------------------------------------------------------------------------
if(NOT DEFINED REPRESENTHW_STANDALONE)
  if(TARGET MLIRIR)
    set(REPRESENTHW_STANDALONE OFF)
  else()
    set(REPRESENTHW_STANDALONE ON)
  endif()
endif()

message(STATUS "RepresentHW standalone: ${REPRESENTHW_STANDALONE}")

# -----------------------------------------------------------------------------
# Language standard
# - Do NOT override the superproject's C++ standard in integrated mode.
# -----------------------------------------------------------------------------
if(REPRESENTHW_STANDALONE)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED YES)
endif()

# -----------------------------------------------------------------------------
# MLIR/LLVM discovery and CMake module setup
# - In integrated mode, assume the superproject already loaded MLIR/LLVM and
#   includes AddLLVM/AddMLIR/TableGen/HandleLLVMOptions as needed.
# -----------------------------------------------------------------------------
if(REPRESENTHW_STANDALONE)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
  option(LLVM_BUILD_TOOLS
         "Build the LLVM tools. If OFF, just generate build targets." ON)

  # Keep these local to standalone mode to avoid polluting a superproject.
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  # lit defaults (standalone only)
  set(LIT_ARGS_DEFAULT "-sv --timeout=30")
  if(MSVC_IDE OR XCODE)
    set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LLVM_LIT_ARGS
      "${LIT_ARGS_DEFAULT}"
      CACHE STRING "Default options for lit")
  set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" FORCE)

  # Output dirs (standalone only)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

  # CMake modules
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
endif()

# Project-local modules (always)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# These includes are safe/needed in standalone mode.
# In integrated mode, they are usually already included by the superproject.
if(REPRESENTHW_STANDALONE)
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  # setup python (standalone only)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include_directories(${PROJECT_SOURCE_DIR}/mlir/include)
include_directories(${PROJECT_BINARY_DIR}/mlir/include)

# -----------------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------------
add_subdirectory(mlir)
# add_subdirectory(tools)
