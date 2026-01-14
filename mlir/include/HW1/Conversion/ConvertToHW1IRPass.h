#ifndef CONVERT_TO_HW1IR_H
#define CONVERT_TO_HW1IR_H

#include "HW1/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace keti {
namespace hw1ir {


std::unique_ptr<mlir::Pass> createLinalgToHW1Pass();



} // namespace hw1ir
} // namespace keti

#endif // CONVERT_TO_HW1IR_H