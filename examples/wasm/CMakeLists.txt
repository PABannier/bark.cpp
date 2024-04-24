set(TARGET libmain)
add_executable(${TARGET} emscripten.cpp)
target_link_libraries(${TARGET} PRIVATE bark)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

unset(EXTRA_FLAGS)

set_target_properties(${TARGET} PROPERTIES LINK_FLAGS " \
    --bind \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE_STRICT=0 \
    -s INITIAL_MEMORY=2000MB \
    -s TOTAL_MEMORY=2000MB \
    -s FORCE_FILESYSTEM=1 \
    -s EXPORTED_RUNTIME_METHODS=\"['out', 'err', 'ccall', 'cwrap']\" \
    ${EXTRA_FLAGS} \
    ")

set(TARGET wasm)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index-tmpl.html ${CMAKE_BINARY_DIR}/examples/${TARGET}/index.html @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/helpers.js      ${CMAKE_BINARY_DIR}/examples/${TARGET}/helpers.js @ONLY)
