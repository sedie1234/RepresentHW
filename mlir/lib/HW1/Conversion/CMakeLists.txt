add_mlir_library(
  HW1ConversionPasses
  ConvertToHW1Pass.cpp
  Passes.cpp

  DEPENDS
  HW1ConversionPassIncGen
#   HW1TransformOpsIncGen
#   HW1TransformPasses
  HW1Dialect


  LINK_COMPONENTS
  Core

  LINK_LIBS
  HW1Dialect
  MLIRIR
  MLIRLinalgUtils
  MLIRLinalgTransforms
  MLIRSupport
  MLIRTransforms
  MLIRPass)
