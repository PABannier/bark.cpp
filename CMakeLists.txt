cmake_minimum_required(VERSION 3.12)
project("bark" C CXX)

set(BARK_BUILD_EXAMPLES ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if (EMSCRIPTEN)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -pthread -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3")
endif()

set(BARK_LIB bark)

add_subdirectory(encodec.cpp)

add_library(${BARK_LIB} STATIC bark.cpp bark.h)

if (BARK_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

target_link_libraries(${BARK_LIB} PUBLIC ggml encodec)
target_include_directories(${BARK_LIB} PUBLIC .)
target_compile_features(${BARK_LIB} PUBLIC cxx_std_11)

if (EMSCRIPTEN)
    set_target_properties(${BARK_LIB} PROPERTIES COMPILE_FLAGS "-msimd128")
endif()

if (GGML_CUBLAS)
    add_compile_definitions(GGML_USE_CUBLAS)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
