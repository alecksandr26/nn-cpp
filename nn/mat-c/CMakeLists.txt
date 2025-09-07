#[[

This is a very simple library for mat multiplications in C
that it will use: OpenMP and SIMD (avx) to accelerate the algorithms

]]

set(SOURCE_DIR src)
set(INCLUDE_DIR include)
set(TEST_DIR test)

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


enable_testing()

file(GLOB TEST_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_DIR}/*.cpp"
)

add_executable(
  mat_tests
  ${TEST_SOURCES}
)

target_link_libraries(
  mat_tests
  PRIVATE
  mat
  GTest::gtest
  GTest::gtest_main
)

add_test(
  NAME mat_tests
  COMMAND mat_tests
)


