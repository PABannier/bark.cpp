set(TARGET bark_small)
add_executable(${TARGET} bark_small.cpp)
target_link_libraries(${TARGET} PRIVATE bark common)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

if(MSVC)
    target_compile_definitions(${TARGET} PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()