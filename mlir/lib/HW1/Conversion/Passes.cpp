#include "HW1/Conversion/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "HW1/Conversion/Passes.h.inc"
}

void keti::hw1ir::registerConversionPasses() { ::registerPasses(); }