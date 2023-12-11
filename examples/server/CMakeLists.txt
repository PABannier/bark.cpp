set(TARGET server)
add_executable(${TARGET} server.cpp httplib.h json.hpp)

install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE bark.cpp ${CMAKE_THREAD_LIBS_INIT})
