cmake_minimum_required(VERSION 3.12) # Don't bump this version for no reason
project("bark.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(BARK_STANDALONE ON)

    # configure project version
    # TODO
else()
    set(BARK_STANDALONE OFF)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(BARK_WASM_SINGLE_FILE "bark: embed WASM inside the generated bark.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()


#
# Option list
#

# general
option(BARK_STATIC                     "bark: static link libraries"                          OFF)
option(BARK_NATIVE                     "bark: enable -march=native flag"                      OFF)
option(BARK_LTO                        "bark: enable link time optimization"                  OFF)

# debug
option(BARK_ALL_WARNINGS               "bark: enable all compiler warnings"                   ON)
option(BARK_ALL_WARNINGS_3RD_PARTY     "bark: enable all compiler warnings in 3rd party libs" OFF)
option(BARK_GPROF                      "bark: enable gprof"                                   OFF)

# sanitizers
option(BARK_SANITIZE_THREAD            "bark: enable thread sanitizer"                        OFF)
option(BARK_SANITIZE_ADDRESS           "bark: enable address sanitizer"                       OFF)
option(BARK_SANITIZE_UNDEFINED         "bark: enable undefined sanitizer"                     OFF)

# instruction set specific
option(BARK_AVX                        "bark: enable AVX"                                     ON)
option(BARK_AVX2                       "bark: enable AVX2"                                    ON)
option(BARK_AVX512                     "bark: enable AVX512"                                  OFF)
option(BARK_AVX512_VBMI                "bark: enable AVX512-VBMI"                             OFF)
option(BARK_AVX512_VNNI                "bark: enable AVX512-VNNI"                             OFF)
option(BARK_FMA                        "bark: enable FMA"                                     ON)
# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(BARK_F16C                   "bark: enable F16C"                                    ON)
endif()

# 3rd party libs
option(BARK_ACCELERATE                      "bark: enable Accelerate framework"               ON)
option(BARK_BLAS                            "bark: use BLAS"                                  OFF)
set(BARK_BLAS_VENDOR "Generic" CACHE STRING "bark: BLAS library vendor")
option(BARK_CUBLAS                          "bark: use cuBLAS"                                OFF)
option(BARK_CUDA_FORCE_DMMV                 "bark: use dmmv instead of mmvq CUDA kernels"     OFF)
set(BARK_CUDA_DMMV_X      "32" CACHE STRING "bark: x stride for dmmv CUDA kernels")
set(BARK_CUDA_MMV_Y        "1" CACHE STRING "bark: y block size for mmv CUDA kernels")
option(BARK_CUDA_DMMV_F16                   "bark: use 16 bit floats for dmmv CUDA kernels"   OFF)
option(BARK_CLBLAST                         "bark: use CLBlast"                               OFF)
option(BARK_METAL                           "bark: use Metal"                                 OFF)

option(BARK_BUILD_TESTS                "bark: build tests"    ${BARK_STANDALONE})

#
# Build info header
#

# Generate initial build-info.h
include(${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-info.cmake)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.git")

    # Is git submodule
    if(NOT IS_DIRECTORY "${GIT_DIR}")
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK)
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK})
        set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${REAL_GIT_DIR}")
    endif()

    # Add a custom target for build-info.h
    add_custom_target(BUILD_INFO ALL DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/build-info.h")

    # Add a custom command to rebuild build-info.h when .git/index changes
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/build-info.h"
        COMMENT "Generating build details from Git"
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-info.cmake"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS "${GIT_DIR}/index"
        VERBATIM
    )
else()
    message(WARNING "Git repository not found; to enable automatic generation of build info, make sure Git is installed and the project is a Git repository.")
endif()

#
# Compile flags
#

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (NOT MSVC)
    if (BARK_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif()

    if (BARK_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()

    if (BARK_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif()
endif()

if (APPLE AND BARK_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        set(BARK_EXTRA_LIBS ${BARK_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (BARK_BLAS)
    if (BARK_STATIC)
        set(BLA_STATIC ON)
    endif()
    if ($(CMAKE_VERSION) VERSION_GREATER_EQUAL 3.22)
        set(BLA_SIZEOF_INTEGER 8)
    endif()

    set(BLA_VENDOR ${BARK_BLAS_VENDOR})
    find_package(BLAS)

    if (BLAS_FOUND)
        message(STATUS "BLAS found, Libraries: ${BLAS_LIBRARIES}")

        if ("${BLAS_INCLUDE_DIRS}" STREQUAL "")
            # BLAS_INCLUDE_DIRS is missing in FindBLAS.cmake.
            # see https://gitlab.kitware.com/cmake/cmake/-/issues/20268
            find_package(PkgConfig REQUIRED)
            if (${BARK_BLAS_VENDOR} MATCHES "Generic")
                pkg_check_modules(DepBLAS REQUIRED blas)
            elseif (${BARK_BLAS_VENDOR} MATCHES "OpenBLAS")
                pkg_check_modules(DepBLAS REQUIRED openblas)
            elseif (${BARK_BLAS_VENDOR} MATCHES "FLAME")
                pkg_check_modules(DepBLAS REQUIRED blis)
            elseif (${BARK_BLAS_VENDOR} MATCHES "ATLAS")
                pkg_check_modules(DepBLAS REQUIRED blas-atlas)
            elseif (${BARK_BLAS_VENDOR} MATCHES "FlexiBLAS")
                pkg_check_modules(DepBLAS REQUIRED flexiblas_api)
            elseif (${BARK_BLAS_VENDOR} MATCHES "Intel")
                # all Intel* libraries share the same include path
                pkg_check_modules(DepBLAS REQUIRED mkl-sdl)
            elseif (${BARK_BLAS_VENDOR} MATCHES "NVHPC")
                # this doesn't provide pkg-config
                # suggest to assign BLAS_INCLUDE_DIRS on your own
                if ("${NVHPC_VERSION}" STREQUAL "")
                    message(WARNING "Better to set NVHPC_VERSION")
                else()
                    set(DepBLAS_FOUND ON)
                    set(DepBLAS_INCLUDE_DIRS "/opt/nvidia/hpc_sdk/${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}/${NVHPC_VERSION}/math_libs/include")
                endif()
            endif()
            if (DepBLAS_FOUND)
                set(BLAS_INCLUDE_DIRS ${DepBLAS_INCLUDE_DIRS})
            else()
                message(WARNING "BLAS_INCLUDE_DIRS neither been provided nor been automatically"
                " detected by pkgconfig, trying to find cblas.h from possible paths...")
                find_path(BLAS_INCLUDE_DIRS
                    NAMES cblas.h
                    HINTS
                        /usr/include
                        /usr/local/include
                        /usr/include/openblas
                        /opt/homebrew/opt/openblas/include
                        /usr/local/opt/openblas/include
                        /usr/include/x86_64-linux-gnu/openblas/include
                )
            endif()
        endif()

        message(STATUS "BLAS found, Includes: ${BLAS_INCLUDE_DIRS}")
        add_compile_options(${BLAS_LINKER_FLAGS})
        add_compile_definitions(GGML_USE_OPENBLAS)
        if (${BLAS_INCLUDE_DIRS} MATCHES "mkl" AND (${BARK_BLAS_VENDOR} MATCHES "Generic" OR ${BARK_BLAS_VENDOR} MATCHES "Intel"))
            add_compile_definitions(GGML_BLAS_USE_MKL)
        endif()
        set(BARK_EXTRA_LIBS ${BARK_EXTRA_LIBS} ${BLAS_LIBRARIES})
        set(BARK_EXTRA_INCLUDES ${BARK_EXTRA_INCLUDES} ${BLAS_INCLUDE_DIRS})

    else()
        message(WARNING "BLAS not found, please refer to "
        "https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors"
        " to set correct BARK_BLAS_VENDOR")
    endif()
endif()

if (BARK_CUBLAS)
    cmake_minimum_required(VERSION 3.17)

    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")

        enable_language(CUDA)

        set(GGML_SOURCES_CUDA ggml-cuda.cu ggml-cuda.h)

        add_compile_definitions(GGML_USE_CUBLAS)
        if (BARK_CUDA_FORCE_DMMV)
            add_compile_definitions(GGML_CUDA_FORCE_DMMV)
        endif()
        add_compile_definitions(GGML_CUDA_DMMV_X=${BARK_CUDA_DMMV_X})
        add_compile_definitions(GGML_CUDA_MMV_Y=${BARK_CUDA_MMV_Y})
        if (DEFINED BARK_CUDA_DMMV_Y)
            add_compile_definitions(GGML_CUDA_MMV_Y=${BARK_CUDA_DMMV_Y}) # for backwards compatibility
        endif()
        if (BARK_CUDA_DMMV_F16)
            add_compile_definitions(GGML_CUDA_DMMV_F16)
        endif()
        add_compile_definitions(K_QUANTS_PER_ITERATION=${BARK_CUDA_KQUANTS_ITER})

        if (BARK_STATIC)
            set(BARK_EXTRA_LIBS ${BARK_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
        else()
            set(BARK_EXTRA_LIBS ${BARK_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if (BARK_CUDA_DMMV_F16)
            set(CMAKE_CUDA_ARCHITECTURES "60;61") # needed for f16 CUDA intrinsics
        else()
            set(CMAKE_CUDA_ARCHITECTURES "52;61") # lowest CUDA 12 standard + lowest for integer intrinsics
        endif()
    endif()
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()

if (BARK_METAL)
    find_library(FOUNDATION_LIBRARY         Foundation              REQUIRED)
    find_library(METAL_FRAMEWORK            Metal                   REQUIRED)
    find_library(METALKIT_FRAMEWORK         MetalKit                REQUIRED)
    find_library(METALPERFORMANCE_FRAMEWORK MetalPerformanceShaders REQUIRED)

    set(GGML_SOURCES_METAL ggml-metal.m ggml-metal.h)

    add_compile_definitions(GGML_USE_METAL)
    add_compile_definitions(GGML_METAL_NDEBUG)

    # get full path to the file
    #add_compile_definitions(GGML_METAL_DIR_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/")

    # copy ggml-metal.metal to bin directory
    configure_file(ggml-metal.metal bin/ggml-metal.metal COPYONLY)

    set(BARK_EXTRA_LIBS ${BARK_EXTRA_LIBS}
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${METALPERFORMANCE_FRAMEWORK}
        )
endif()

if (BARK_CLBLAST)
    find_package(CLBlast)
    if (CLBlast_FOUND)
        message(STATUS "CLBlast found")

        set(GGML_SOURCES_OPENCL ggml-opencl.cpp ggml-opencl.h)

        add_compile_definitions(GGML_USE_CLBLAST)

        set(BARK_EXTRA_LIBS ${BARK_EXTRA_LIBS} clblast)
    else()
        message(WARNING "CLBlast not found")
    endif()
endif()

if (BARK_ALL_WARNINGS)
    if (NOT MSVC)
        set(c_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wdouble-promotion
            -Wshadow
            -Wstrict-prototypes
            -Wpointer-arith
            -Wmissing-prototypes
        )
        set(cxx_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wno-unused-function
            -Wno-multichar
        )
    else()
        # todo : msvc
    endif()

    add_compile_options(
            "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )

endif()

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

if (BARK_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (BARK_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (BARK_GPROF)
        add_compile_options(-pg)
    endif()
    if (BARK_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            # Raspberry Pi 2
            add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            add_compile_options(-mfp16-format=ieee -mno-unaligned-access)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (BARK_AVX512)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (BARK_AVX512_VBMI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
            endif()
            if (BARK_AVX512_VNNI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
        elseif (BARK_AVX2)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        elseif (BARK_AVX)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
        endif()
    else()
        if (BARK_F16C)
            add_compile_options(-mf16c)
        endif()
        if (BARK_FMA)
            add_compile_options(-mfma)
        endif()
        if (BARK_AVX)
            add_compile_options(-mavx)
        endif()
        if (BARK_AVX2)
            add_compile_options(-mavx2)
        endif()
        if (BARK_AVX512)
            add_compile_options(-mavx512f)
            add_compile_options(-mavx512bw)
        endif()
        if (BARK_AVX512_VBMI)
            add_compile_options(-mavx512vbmi)
        endif()
        if (BARK_AVX512_VNNI)
            add_compile_options(-mavx512vnni)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    add_compile_options(-mcpu=native -mtune=native)
    #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
else()
    message(STATUS "Unknown architecture")
endif()

#
# Build libraries
#

add_library(ggml OBJECT
            ggml.c
            ggml.h
            ${GGML_SOURCES_CUDA}
            ${GGML_SOURCES_OPENCL}
            ${GGML_SOURCES_METAL}
            ${GGML_SOURCES_EXTRA}
            )

target_include_directories(ggml PUBLIC . ${BARK_EXTRA_INCLUDES})
target_compile_features(ggml PUBLIC c_std_11) # don't bump
target_link_libraries(ggml PUBLIC Threads::Threads ${BARK_EXTRA_LIBS})

add_library(ggml_static STATIC $<TARGET_OBJECTS:ggml>)
if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(ggml_shared SHARED $<TARGET_OBJECTS:ggml>)
    target_link_libraries(ggml_shared PUBLIC Threads::Threads ${BARK_EXTRA_LIBS})
    install(TARGETS ggml_shared LIBRARY)
endif()

add_library(bark
            bark.cpp
            bark.h
            bark-util.h
            encodec.cpp
            encodec.h
            )

target_include_directories(bark PUBLIC .)
target_compile_features(bark PUBLIC cxx_std_11) # don't bump
target_link_libraries(bark PRIVATE
    ggml
    ${BARK_EXTRA_LIBS}
    )

if (BUILD_SHARED_LIBS)
    set_target_properties(bark PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(bark PRIVATE BARK_SHARED BARK_BUILD)
    if (BARK_METAL)
        set_target_properties(bark PROPERTIES RESOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ggml-metal.metal")
    endif()
    install(TARGETS bark LIBRARY)
endif()

include(GNUInstallDirs)
install(
    FILES convert.py
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR})

#
# programs, examples and tests
#

if (BARK_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif ()
