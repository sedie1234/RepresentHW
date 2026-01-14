#ifndef HW1IR_INITALL_H
#define HW1IR_INITALL_H

#include "mlir/IR/Dialect.h"

namespace keti {
namespace hw1ir {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace hw1ir
} // namespace keti

#endif // HW1IR_INITALL_H