#[[

This is a very simple library for mat multiplications in C
that it will use: OpenMP and SIMD (avx) to accelerate the algorithms

]]

set(SOURCE_DIR src)

set(INCLUDE_DIR include)

file(GLOB SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_DIR}/*.c"
)


add_library(
  mat
  STATIC
  ${SOURCES}
)

target_include_directories(
  mat
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}
)

