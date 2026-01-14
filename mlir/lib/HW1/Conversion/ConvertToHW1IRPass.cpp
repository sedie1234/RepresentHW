#include "HW1/Conversion/ConvertToHW1IRPass.h"
#include "HW1/Dialect/HW1IR/HW1IRDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/ComposeSubView.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"


using namespace mlir;

namespace keti {
namespace hw1ir {


// pass structure
struct LinalgToHW1Pass
    : public keti::hw1ir::impl::LinalgToHW1Base<LinalgToHW1Pass> {
    LinalgToHW1Pass() = default;
    LinalgToHW1Pass(const LinalgToHW1Pass &pass) {}

    void runOnOperation() override;
}


// conversion : rewrite patterns structure
class LinalgMatmulToHW1Pattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                  PatternRewriter &rewriter) const override {

        Value inputA = matmulOp.getInputs()[0];
        Value inputB = matmulOp.getInputs()[1];
        Value outputC = matmulOp.getOutputs()[0];
    
        rewriter.replaceOpWithNewOp<keti::hw1ir::MatmulOp>(
            matmulOp,
            matmulOp.getResult(0).getType(),
            inputA,
            inputB
        );

        return success();
    }
}

void LinalgToHW1Pass::runOnOperation() {
    auto module = getOperation();
    auto context = module.getContext();

    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    RewritePatternSet patterns(context);
    
    // Add patterns to convert Linalg ops to HW1IR ops here
    patterns.add<LinalgMatmulToHW1Pattern>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
    }

    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));
}

}
}