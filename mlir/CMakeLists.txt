add_subdirectory(lib)
add_subdirectory(include)
# add_subdirectory(test)

if(NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(
    DIRECTORY include/HW1
    DESTINATION include
    COMPONENT hw1-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT")

#   install(
#     DIRECTORY ${PROJECT_BINARY_DIR}/include/HW2
#     DESTINATION include
#     COMPONENT hw2-headers
#     FILES_MATCHING
#     PATTERN "*.def"
#     PATTERN "*.h"
#     PATTERN "*.gen"
#     PATTERN "*.inc"
#     PATTERN "*.td"
#     PATTERN "CMakeFiles" EXCLUDE
#     PATTERN "config.h" EXCLUDE)

  if(NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-hw1-headers DEPENDS hw1-headers COMPONENT
                             hw1-headers)
  endif()
endif()