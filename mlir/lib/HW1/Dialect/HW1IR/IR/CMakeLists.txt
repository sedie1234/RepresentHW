add_mlir_dialect_library(
  HW1IRDialect
  HW1IRDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/HW1/Dialect/HW1IR

  DEPENDS
  MLIRHW1IROpsIncGen
  MLIRHW1IRDialectIncGen
  MLIRHW1OpInterfacesIncGen
  MLIRLinalgDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRDialect
  
  )

  
target_include_directories(HW1IRDialect PUBLIC
  ${PROJECT_SOURCE_DIR}/mlir/include
  ${PROJECT_BINARY_DIR}/mlir/include
)