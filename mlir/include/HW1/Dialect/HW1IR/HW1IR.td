#ifndef HW1IR_OPS
#define HW1IR_OPS

include "mlir/IR/OpBase.td"
include "HW1/Dialect/HW1IR/HW1IROpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
// include "mlir/IR/AffineMap.td"

// base operation
class hw1ir_Op<string mnemonic, list<Trait> traits = []> :
    Op<hw1ir_Dialect, mnemonic, traits>;


// operation definitions


def hw1ir_MatmulOp : hw1ir_Op<"matmul", [Pure]>{
    let summary = "Matrix Multiplication Operation";
    let arguments = (ins
        AnyTypeOf<[AnyMemRef, AnyTensor]>:$A,
        AnyTypeOf<[AnyMemRef, AnyTensor]>:$B,
        DefaultValuedAttr<I32Attr, "0">:$mode
    );
    // 결과를 다시 1개로 정의합니다 (Tensor 모드 대응)
    let results = (outs AnyTypeOf<[AnyMemRef, AnyTensor]>:$C);
    
    let assemblyFormat = [{
        $A `,` $B attr-dict `:` type($A) `,` type($B) `->` type($C)
    }];
}
 

def hw1ir_ReluOp : hw1ir_Op<"relu", [Pure]>{
    let summary = "ReLU Activation Operation";
    let description = [{
        This operation applies the ReLU activation function to the input tensor.
        It is just for test.
    }];
    let arguments = (ins
        AnyTypeOf<[AnyMemRef, AnyTensor]>:$input
    );
    let results = (outs AnyTypeOf<[AnyMemRef, AnyTensor]>:$output);
    let assemblyFormat = [{
        $input attr-dict `:` type($input) `->` type($output)
    }];
}


#endif // HW1IR_OPS