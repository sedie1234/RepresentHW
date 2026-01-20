#include "HW1/Dialect/HW1IR/HW1IRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/TypeSwitch.h"

#include <iostream>

using namespace mlir;

#include "Dialect/HW1IR/HW1IRDialect.cpp.inc"

namespace keti {

void hw1ir::hw1irDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/HW1IR/HW1IROps.cpp.inc"
      >();
}

} // namespace keti
