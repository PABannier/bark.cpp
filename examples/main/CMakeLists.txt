set(TARGET main)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE bark common)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

if(MSVC)
    target_compile_definitions(${TARGET} PRIVATE -D_CRT_SECURE_NO_WARNINGS=1)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10485760")
endif()