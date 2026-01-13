add_mlir_dialect_library(
  HW1IRDialect
  HW1IRDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/HW1/Dialect/HW1IR

  DEPENDS
  MLIRHW1IRIncGen
#   MLIRHW1IROpInterfacesIncGen
  MLIRLinalgDialect

  LINK_LIBS PUBLIC
  MLIRIR)

  DEPENDS
  MLIRHW1IRIncGen
#   MLIRHW1IROpInterfacesIncGen
  MLIRLinalgDialect

  LINK_LIBS PUBLIC
  MLIRIR)
