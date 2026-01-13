set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(HW1ConversionPassIncGen)

add_mlir_doc(Passes HW1ConversionPasses ./ -gen-pass-doc)