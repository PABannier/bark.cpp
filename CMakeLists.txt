cmake_minimum_required(VERSION 3.12) 
project("bark.cpp" C CXX)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(BARK_STANDALONE ON)
else()
    set(BARK_STANDALONE OFF)
endif()

option(BARK_BUILD_TESTS                "bark: build tests"    ${BARK_STANDALONE})
option(BARK_BUILD_EXAMPLES             "bark: build examples" ${BARK_STANDALONE})

# Build libraries

add_subdirectory(ggml)

set(BARK_LIB bark.cpp)

add_library(
        ${BARK_LIB}
        bark
        bark.cpp
        bark.h
        bark-util.h
        encodec.cpp
        encodec.h
)

target_link_libraries(${BARK_LIB} PUBLIC ggml)
target_include_directories(${BARK_LIB} PUBLIC .)
target_compile_features(${BARK_LIB} PUBLIC cxx_std_11)

if (BARK_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (BARK_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif ()
