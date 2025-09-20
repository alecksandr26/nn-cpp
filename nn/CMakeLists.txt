#[[

For compiling this project:

mkdir -p nn/build/ && cd nn/build/ && cmake .. && make

For testing this project:

cd nn/build/ && make && ./nn_tests
cd nn/build/ && make && ./mat-c/mat_tests
]]

cmake_minimum_required(VERSION 3.10)

project(
  nn
  VERSION 1.0.0
  DESCRIPTION "A nerual network library in C/C++"
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -fsanitize=address-fsanitize=address
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "-Wall -Wextra -pedantic ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb")
set(CMAKE_C_FLAGS_DEBUG "-ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -ffast-math -march=native")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG -ffast-math -march=native")

if(DEFINED CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} ")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_RELEASE} ${CMAKE_C_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
endif()

message("-- CXX Compiler Flags: ${CMAKE_CXX_FLAGS}")
message("-- C Compiler Flags: ${CMAKE_C_FLAGS}")

# Enabling the testing
include(FetchContent)

cmake_policy(SET CMP0135 NEW)

# Download and configure GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

# Prevent GoogleTest from installing itself
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

set(SOURCE_DIR src)
set(INCLUDE_DIR include)
set(TEST_DIR test)

file(GLOB SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_DIR}/*.cpp"
)

# Add the library
add_library(
  nn
  SHARED
  ${SOURCES}
)

# Target the library
target_include_directories(
  nn
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}
)

# Add the library and link it
add_subdirectory(mat-c)
target_link_libraries(
  nn
  PRIVATE
  mat
)

enable_testing()

file(GLOB TEST_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_DIR}/test_*.cpp"
)

# Link the executable
add_executable(
  nn_tests
  ${TEST_SOURCES}
)

target_link_libraries(
  nn_tests
  PRIVATE
  nn
  GTest::gtest
  GTest::gtest_main
)

add_test(
  NAME nn_tests
  COMMAND mat_tests
)


