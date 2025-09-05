#[[

For compiling this project:

mkdir -p build/ && cd build/ && cmake ../nn/ && make

]]

cmake_minimum_required(VERSION 3.10)

project(
  nn
  VERSION 1.0.0
  DESCRIPTION "A nerual network library in C/C++"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCE_DIR src)
set(INCLUDE_DIR include)

file(GLOB SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_DIR}/*.cpp"
)

add_library(
  nn
  SHARED
  ${SOURCES}
)


target_include_directories(
  nn
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}
)


# Include the mat library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mat-c)

target_link_libraries(
  nn
  PRIVATE
  mat
)



