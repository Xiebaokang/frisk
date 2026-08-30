if(NOT TARGET llvm_gtest)
  add_subdirectory(
    ${LLVM_MAIN_SRC_DIR}/../third-party/unittest
    ${CMAKE_CURRENT_BINARY_DIR}/third-party/unittest
    EXCLUDE_FROM_ALL
  )
endif()

add_custom_target(FriskUnitTests)

add_subdirectory(Dialect/Frisk)
