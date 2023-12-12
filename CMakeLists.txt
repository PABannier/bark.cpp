cmake_minimum_required(VERSION 3.12)
project("bark" C CXX)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(BARK_STANDALONE ON)
else()
    set(BARK_STANDALONE OFF)
endif()

option(BARK_BUILD_TESTS                "bark: build tests"    ${BARK_STANDALONE})
option(BARK_BUILD_EXAMPLES             "bark: build examples" ${BARK_STANDALONE})

# Build libraries

set(BARK_LIB bark)

add_subdirectory(ggml)
# add_subdirectory(encodec.cpp)

add_library(${BARK_LIB} bark bark.cpp bark.h)

if (BARK_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (BARK_BUILD_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif ()

target_link_libraries(${BARK_LIB} PUBLIC ggml)
# target_link_libraries(${BARK_LIB} PUBLIC encodec)
target_include_directories(${BARK_LIB} PUBLIC .)
target_compile_features(${BARK_LIB} PUBLIC cxx_std_11)
