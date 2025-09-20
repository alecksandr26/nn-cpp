#[[

For compiling this project:

mkdir -p demos/demo-1/build/ && cd demos/demo-1/build/ && cmake ..  -Wno-dev && make

For running this project

cd demos/demo-1/ && ./build/demo1

]]

cmake_minimum_required(VERSION 3.10)

project(
  demo1
  VERSION 1.0.0
  DESCRIPTION "A simple UI demoa of the library"
  LANGUAGES CXX
)

set(CMAKE_AUTOMOC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

add_subdirectory(../../nn/ nn)

set(SOURCE_DIR src)

# Find Qt6 Widgets
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Charts)


#[[
cd demos/demos-1/ && ./build/demo1
]]
add_executable(${PROJECT_NAME}
  ${SOURCE_DIR}/main.cpp
  ${SOURCE_DIR}/chart.cpp
  ${SOURCE_DIR}/chart.hpp
  ${SOURCE_DIR}/controls.cpp
  ${SOURCE_DIR}/controls.hpp
  ${SOURCE_DIR}/trainer.cpp
  ${SOURCE_DIR}/trainer.hpp
)

# Link Qt6 Widgets
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
  Qt6::Core
  Qt6::Widgets
  Qt6::Charts
)

# Link the neural network library
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
  nn
)

