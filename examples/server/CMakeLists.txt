set(TARGET server)
add_executable(${TARGET} server.cpp httplib.h json.hpp)

install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE bark ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()
