cmake_minimum_required(VERSION 3.9)

project(RCE3D LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CUDA_FLAGS "-gencode arch=compute_61,code=sm_61")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(UNIX)
    find_package(SDL2 REQUIRED)
    find_package(SDL2_image REQUIRED)
    find_package(SDL2_ttf REQUIRED)
    find_package(SDL2_mixer REQUIRED)
    include_directories(
        ${SDL2_INCLUDE_DIRS} 
        ${SDL2_IMAGE_INCLUDE_DIR} 
        ${SDL2_TTF_INCLUDE_DIR}
        ${SDL2_MIXER_INCLUDE_DIR}
    )
endif()

if(WIN32)
    include_directories("${CMAKE_SOURCE_DIR}/SDL2/include")
    link_directories("${CMAKE_SOURCE_DIR}/SDL2/lib/")
endif()

file(GLOB_RECURSE SRC_CPP_FILES "src/*.cpp")
file(GLOB_RECURSE SRC_CU_FILES "src/*.cu")
file(GLOB_RECURSE HEADER_FILES "src/*.hpp", "src/*.cuh")

add_executable(
    ${PROJECT_NAME}
    ${SRC_CPP_FILES}
    ${SRC_CU_FILES}
    ${HEADER_FILES}
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
target_compile_features(
    ${PROJECT_NAME}
    PUBLIC cxx_std_11
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)

if(UNIX)
    target_link_libraries(
        ${PROJECT_NAME}
        ${SDL2_LIBRARY} 
        ${SDL2_IMAGE_LIBRARIES}
        ${SDL2_TTF_LIBRARIES}
        ${SDL2_MIXER_LIBRARIES}
    )
endif()

if(WIN32)
    target_link_libraries(
        ${PROJECT_NAME}
        SDL2.lib
        SDL2main.lib
        SDL2_image.lib
        SDL2_mixer.lib
        SDL2_ttf.lib
    )
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

