cmake_minimum_required(VERSION 3.16)
project(omr CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(omr
    src/main.cpp
    src/core/CornerFinder.cpp
    src/core/PerspectiveCorrector.cpp
)

target_link_libraries(omr ${OpenCV_LIBS})
