cmake_minimum_required(VERSION 3.21)
project(Neural_network C)

set(CMAKE_C_STANDARD 11)
add_executable(Neural_network main.c NNStruct.c NNStruct.h)
target_link_libraries(Neural_network m)
