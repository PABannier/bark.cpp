set(TARGET server)
add_executable(${TARGET} server.cpp httplib.h json.hpp)

install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE bark common ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
    if(MSVC)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10485760")
    endif()
endif()
