#ifndef HW1IR_CONVERSION_PASSES
#define HW1IR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToHW1 : Pass<"linalg-to-hw1", "ModuleOp">{
    let summary = "Convert Linalg operations to HW1IR dialect";
    let constructor = "keti::hw1ir::createLinalgToHW1Pass()";
    let description = [{
        This pass converts Linalg operations to HW1IR dialect operations.
    }];
}

#endif // HW1IR_CONVERSION_PASSES