set(TARGET main)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE bark common)
target_compile_features(${TARGET} PRIVATE cxx_std_11)
