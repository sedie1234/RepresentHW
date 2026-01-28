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


namespace keti {
namespace hw1ir {

#define GEN_PASS_DEF_LINALGTOHW1
#include "Conversion/Passes.h.inc"

struct LinalgToHW1Pass
    : public keti::hw1ir::impl::LinalgToHW1Base<LinalgToHW1Pass> {
    LinalgToHW1Pass() = default;
    
    // [수정 1] 부모 클래스의 복사 생성자를 반드시 호출해야 합니다.
    LinalgToHW1Pass(const LinalgToHW1Pass &pass) : LinalgToHW1Base(pass) {}

    void runOnOperation() override;
};

class LinalgMatmulToHW1Pattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
        // 1. 입력 추출
        Value inputA = matmulOp.getInputs()[0];
        Value inputB = matmulOp.getInputs()[1];
        
        // 2. 결과 타입 결정
        Type resultType = matmulOp->getNumResults() > 0 
                        ? matmulOp->getResult(0).getType() 
                        : matmulOp.getOutputs()[0].getType();

        // 3. 연산 생성 (가장 표준적인 build 함수 호출)
        // 인자 순서: rewriter, location, resultType, operands..., attributes...
        auto newOp = rewriter.create<keti::hw1ir::MatmulOp>(
            matmulOp.getLoc(), 
            resultType, 
            inputA, 
            inputB,
            rewriter.getI32IntegerAttr(0) // mode
        );

        // 4. 결과 연결 및 기존 Op 제거
        if (matmulOp->getNumResults() > 0) {
            rewriter.replaceOp(matmulOp, newOp.getResult());
        } else {
            rewriter.eraseOp(matmulOp);
        }

        return success();
    }
};

void LinalgToHW1Pass::runOnOperation() {
    auto module = getOperation();
    auto context = module.getContext();

    
    llvm::errs() << "\n\n========== [Entry Module]  ==========\n";
    module->print(llvm::errs());
    llvm::errs() << "\n===================================\n\n";


    RewritePatternSet patterns(context);
    patterns.add<LinalgMatmulToHW1Pattern>(context);

    // applyPatternsGreedily는 전체 모듈보다는 모듈 내의 함수 단위로 도는 것이 안정적입니다.
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
    }

    // [강제 종료] 뒤쪽 VM 변환 로직이 실행되어 크래시가 나기 전에 여기서 결과를 출력하고 끝냅니다.
    llvm::errs() << "\n\n=== [SUCCESS] IR 변환 결과 확인 ===\n";
    module->print(llvm::errs());
    llvm::errs() << "\n===================================\n\n";
    
    // 컴파일러를 정상 종료 상태로 리턴시키지 않고 프로세스 자체를 죽여서 
    // 뒤쪽의 serialization (undefined reference) 단계를 원천 차단합니다.
    exit(0); 
}

std::unique_ptr<mlir::Pass> createLinalgToHW1Pass() {
  return std::make_unique<LinalgToHW1Pass>();
}

} // namespace hw1ir
} // namespace keti