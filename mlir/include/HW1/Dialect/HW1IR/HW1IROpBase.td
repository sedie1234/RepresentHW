#ifndef HW1IR_OPS_BASE
#define HW1IR_OPS_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def hw1ir_Dialect : Dialect {
  let name = "hw1ir";
  let cppNamespace = "keti::hw1ir";
  let description = [{
    The `hw1ir` dialect is test IR. 
  }];
  // let useDefaultPrinterParser = 1;
}

def hw1ir_AnyScalarOrTensor : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                           "any scalar or tensor type">;


def L1: I32EnumAttrCase<"L1", 2>;
def L2: I32EnumAttrCase<"L2", 1>;
def L3: I32EnumAttrCase<"L3", 0>;

//memory space enum attr definition
def MemorySpace: I32EnumAttr<"MemorySpace", "HW1IR Memory Space IDs",
  [L1, L2, L3]> {

  let cppNamespace = "keti::hw1ir";
}

def hw1ir_AsyncToken : DialectType<
    hw1ir_Dialect, CPred<"llvm::isa<keti::hw1ir::AsyncTokenType>($_self)">, "async token type">,
             BuildableType<"keti::hw1ir::AsyncTokenType::get($_builder.getContext())">;

/*****************************************************************************/
// interface는 operation이 어떤 기능이 있는지 보장해 줌. 
// -> op는 어떤 역할을 수행하는지, op가 가지고 있는 interface는 무엇을 할 수 있는지를 나타냄
// 예를 들어 memcpy interface를 가졌다면, memcpy operation이라는 것을 보장해 줌.
// 그러므로 메모리 복사가 일어나는 operation이고, memcpy interface가 제공하는 method들을 사용할 수 있음.
// operation을 호출할 경우, if( op == matmulOp ? )로 호출할 수도 있지만, 
// if( op has memcpy interface ? )로 호출할 수도 있음.
/*****************************************************************************/

// op interfaces

def hw1ir_AsyncOpInterface : OpInterface<"AsyncOpInterface"> {
  let description = [{
    Interface for HW1IR operations that execute asynchronously.
    It is just for test.
    }];
    let cppNamespace = "keti::hw1ir";
    let methods = [
    InterfaceMethod<[{
        Query the operands that represent async dependency tokens.
      }],
      "OperandRange", "getAsyncDependencies", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncDependencies();
//       }]>,
//     InterfaceMethod<[{
//         Adds a new token to the list of async dependencies.
//       }],
//       "void", "addAsyncDependency", (ins "Value":$token),
//       [{}], [{
//         ::keti::hw1ir::addAsyncDependency(this->getOperation(), token);
//       }]>,
//     InterfaceMethod<[{
//         Query whether the operation has an async token result.
//       }],
//       "bool", "hasAsyncTokenResult", (ins), [{}], [{
//         ConcreteOp op = cast<ConcreteOp>(this->getOperation());
//         return op.hasAsyncTokenResult();
//       }]>,
    InterfaceMethod<[{
        Get the async token result of the operation.
      }],
      "Value", "getAsyncTokenResult", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAsyncTokenResult();
      }]>
  ];
}

def hw1ir_MemcpyInterface : OpInterface<"MemcpyInterface"> {
    let description = [{
      Interface for HW1IR memory copy operations.
      It is just for test.
    }];
    let cppNamespace = "keti::hw1ir";
    let methods = [
        InterfaceMethod<[{
            "description"
            }],
            "Value",
            "getSrcMemref"
        >,
        InterfaceMethod<[{
            "description"
            }],
            "Value",
            "getDstMemref"
        >,
        InterfaceMethod<[{
            "description"
            }],
            "Value",
            "getSize"
        >
    ];
}

#endif // HW1IR_OPS_BASE