# ##############################################################################
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: MIT
# ##############################################################################
cmake_minimum_required(VERSION 3.10.2)
project(vpl_demo)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS -pthread)
# Default install places 64 bit runtimes in the environment, so we want to do a
# 64 bit build by default.
if(WIN32)
  message(FATAL_ERROR "Windows not supported by this sample.")
endif()

find_package(PkgConfig REQUIRED)
include(CMakeParseArguments)

macro(add_demo)
  set(oneValueArgs NAME SRC)
  cmake_parse_arguments(VPL_DEMO "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCES main.cpp)
  message("-- demo name  = ${VPL_DEMO_NAME}")
  # Set default build type to RelWithDebInfo if not specified
  if(NOT CMAKE_BUILD_TYPE)
    message(
      STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set(CMAKE_BUILD_TYPE
        "RelWithDebInfo"
        CACHE
          STRING
          "Choose build type from: None Debug Release RelWithDebInfo MinSizeRel"
          FORCE)
  endif()
  aux_source_directory(${VPL_DEMO_SRC} SRC)
  add_executable(${VPL_DEMO_NAME} ${SOURCES} ${SRC})
  find_package(VPL REQUIRED)
  target_link_libraries(${VPL_DEMO_NAME} VPL::dispatcher)
  include_directories(../)

  if(UNIX)
    set(LIBVA_SUPPORT
        ON
        CACHE BOOL "Enable hardware support.")
    if(LIBVA_SUPPORT)
      # note: pkg-config version for libva is *API* version

      pkg_check_modules(PKG_LIBVA libva>=1.2)
      if(PKG_LIBVA_FOUND)
        target_compile_definitions(${VPL_DEMO_NAME} PUBLIC -DLIBVA_SUPPORT)
        target_link_libraries(${VPL_DEMO_NAME}  ${PKG_LIBVA_LIBRARIES})
        target_include_directories(${VPL_DEMO_NAME} PUBLIC ${PKG_LIBVA_INCLUDEDIR})
      else()
        message(
          SEND_ERROR
            "libva not found: set LIBVA_SUPPORT=OFF to build ${TARGET} without libva support"
        )
      endif()

      pkg_check_modules(PKG_LIBVADRM libva-drm>=1.2)
      if(PKG_LIBVADRM_FOUND)
        target_compile_definitions(${VPL_DEMO_NAME} PUBLIC -DLIBVA_SUPPORT)
        target_link_libraries(${VPL_DEMO_NAME} ${PKG_LIBVADRM_LIBRARIES})
        target_include_directories(${VPL_DEMO_NAME} PUBLIC ${PKG_LIBVADRM_INCLUDEDIR})
      else()
        message(
          SEND_ERROR
            "libva-drm not found: set LIBVA_SUPPORT=OFF to build ${TARGET} without libva support"
        )
      endif()
    else()
      message(STATUS "Building ${TARGET} without hardware support")
    endif()
  endif()

  find_package(OpenVINO REQUIRED)
  if(OpenVINO_FOUND)
    target_link_libraries(${VPL_DEMO_NAME} openvino::runtime)
  else()
    message(SEND_ERROR "OpenVINO not found")
  endif()

  find_package(OpenCL)
  if(OpenCL_FOUND)
    target_include_directories(${VPL_DEMO_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
  else()
    message(
      "OpenCL not found with find_package(OpenCL), using backup approach to find OpenCL library."
    )
    find_path(
      OpenCL_LIBRARY_PATH libOpenCL.so.1
      PATHS $ENV{ONEAPI_ROOT}/compiler/latest/linux/lib /usr/lib/x86_64-linux-gnu
      NO_DEFAULT_PATH)
    if(OpenCL_LIBRARY_PATH)
      set(OpenCL_LIBRARIES ${OpenCL_LIBRARY_PATH}/libOpenCL.so.1)
      set(OpenCL_FOUND true)
    endif()

  endif()

  if(OpenCL_FOUND)
    message("using OpenCL library ${OpenCL_LIBRARIES}")
    target_link_libraries(${VPL_DEMO_NAME} ${OpenCL_LIBRARIES})
  else()
    message(SEND_ERROR "OpenCL not found")
  endif()

  find_package(gflags)
  if (gflags_FOUND)
    message("using gflags library")
    target_link_libraries(${VPL_DEMO_NAME} gflags)
  else()
    message(SEND_ERROR "gflags not found")
  endif()

endmacro()

add_subdirectory(single_src)
add_subdirectory(multi_src)

SET(CMAKE_BUILD_TYPE "Debug")  
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")