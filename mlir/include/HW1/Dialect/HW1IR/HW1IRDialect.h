//===- AIRDialect.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_HW1IR_DIALECT_H
#define MLIR_HW1IR_DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringRef.h"

#include <map>

using namespace mlir;

namespace keti {
namespace hw1ir {

class AsyncTokenType
    : public Type::TypeBase<AsyncTokenType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
  static constexpr StringLiteral name = "keti.hw1ir.async_token";
};

// void addAsyncDependency(Operation *op, Value token);
// void eraseAsyncDependency(Operation *op, unsigned index);

} // namespace hw1ir
} // namespace keti

// #include "HW1/Dialect/HW1IR/HW1IRDialect.h.inc"
// #include "HW1/Dialect/HW1IR/HW1IREnums.h.inc"
// #include "HW1/Dialect/HW1IR/HW1IROpInterfaces.h.inc"

#include "Dialect/HW1IR/HW1IRDialect.h.inc"
#include "Dialect/HW1IR/HW1IREnums.h.inc"
#include "Dialect/HW1IR/HW1IRInterface.h.inc"

// include TableGen generated Op definitions
#define GET_OP_CLASSES
// #include "HW1/Dialect/HW1IR/HW1IROps.h.inc"

#include "Dialect/HW1IR/HW1IROps.h.inc"

#endif
