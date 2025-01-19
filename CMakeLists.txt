cmake_minimum_required(VERSION 3.10)
project(meta-BAMDP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include(FetchContent)

FetchContent_Declare(
  unordered_dense
  GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
  GIT_TAG        main  # You can specify a specific commit/tag
)


FetchContent_MakeAvailable(unordered_dense)
# Fetch Boost headers
find_package(Boost REQUIRED)
# Define executables
add_executable(meta-BAMDP src/main.cpp src/m_states.cpp src/meta_node.cpp src/meta_policy.cpp)

target_link_libraries(meta-BAMDP unordered_dense::unordered_dense)
target_include_directories(meta-BAMDP PRIVATE ${unordered_dense_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})

