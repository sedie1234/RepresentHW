add_subdirectory(Conversion)
add_subdirectory(Dialect)
# add_subdirectory(Transform)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_library(
  HW1InitAll
  InitAll.cpp

  DEPENDS
  HW1Dialect
  HW1ConversionPasses
#   HW1TransformOpsIncGen
#   HW1TransformPasses

  LINK_LIBS
  HW1ConversionPasses
#   HW1TransformPasses
#   HW1TransformOps
  HW1Dialect
  MLIRSupport
  ${conversion_libs}
  ${dialect_libs})