#include "HW1/InitAll.h"

#include "HW1/Conversion/Passes.h"
#include "HW1/Dialect/HW1IR/HW1IRDialect.h"
#include "HW1/Dialect/HW1IR/HW1IRTransformOps.h"
#include "HW1/Dialect/HW1Rt/HW1RtDialect.h"
#include "air/Transform/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"

void keti::hw1ir::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<keti::hw1ir::HW1IRDialect>();
  keti::hw1ir::registerTransformDialectExtension(registry);
}

void keti::hw1ir::registerAllPasses() {
//   keti::hw1ir::registerTransformPasses();
  keti::hw1ir::registerConversionPasses();
}
