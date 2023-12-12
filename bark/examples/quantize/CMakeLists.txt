set(TARGET quantize)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE bark)
target_compile_features(${TARGET} PRIVATE cxx_std_11)
