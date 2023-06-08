cmake_minimum_required(VERSION 3.0)

project(estimator)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

add_executable(${PROJECT_NAME} "estimator.cpp")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

