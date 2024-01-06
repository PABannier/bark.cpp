add_library(test_utils STATIC common.cpp)
target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(test_utils PRIVATE cxx_std_11)

#
# test-tokenizer

set(TEST_TARGET test-tokenizer)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE bark)
