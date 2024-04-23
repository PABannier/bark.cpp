# First step: build the Bark library for WASM with Emscripten
set(TARGET libmain)
add_executable(${TARGET} emscripten.cpp)
target_link_libraries(${TARGET} PRIVATE bark)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set_target_properties(${TARGET}
    PROPERTIES
        EXPORT_COMPILE_COMMANDS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

unset(EXTRA_FLAGS)

if (BARK_WASM_SINGLE_FILE)
    set(EXTRA_FLAGS "-s SINGLE_FILE=1")
    message(STATUS "Embedding WASM inside main.js")

    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/bin/libmain.js
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/wasm/main.js
    )
endif()

set_target_properties(${TARGET} PROPERTIES LINK_FLAGS " \
    --bind \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE_STRICT=0 \
    -s INITIAL_MEMORY=2000MB \
    -s TOTAL_MEMORY=2000MB \
    -s FORCE_FILESYSTEM=1 \
    -s EXPORTED_RUNTIME_METHODS=\"['print', 'printErr', 'ccall', 'cwrap']\" \
    ${EXTRA_FLAGS} \
    ")

# Second step: build the main.js file

set(TARGET wasm)

set(WASM_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/wasm)
file(MAKE_DIRECTORY ${WASM_OUTPUT_DIR})

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index-tmpl.html ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/index.html @ONLY)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/helpers.js      ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/helpers.js @ONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index-tmpl.html ${WASM_OUTPUT_DIR}/index.html @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/helpers.js      ${WASM_OUTPUT_DIR}/helpers.js @ONLY)
