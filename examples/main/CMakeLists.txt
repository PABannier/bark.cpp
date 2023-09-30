set(BARK_TARGET main)

add_executable(${BARK_TARGET} main.cpp)

install(TARGETS ${BARK_TARGET} RUNTIME)
target_link_libraries(${BARK_TARGET} PRIVATE bark.cpp ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${BARK_TARGET} PRIVATE cxx_std_11)

if(MSVC)
    target_compile_definitions(${BARK_TARGET} PRIVATE -D_CRT_SECURE_NO_WARNINGS=1)
endif()