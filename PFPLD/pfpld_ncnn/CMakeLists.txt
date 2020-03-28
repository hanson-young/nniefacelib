cmake_minimum_required (VERSION 2.6)

project (pfpld_ncnn)
find_package(OpenMP)
if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
add_executable(pfpld_ncnn src/anchor_generator.cpp  src/config.cpp  src/detect.cpp  src/tools.cpp
src/anchor_generator.h src/config.h src/tools.h src/pfpld.id.h)
find_package( OpenCV REQUIRED )
target_link_libraries( pfpld_ncnn ${OpenCV_LIBS} )

set_target_properties(pfpld_ncnn PROPERTIES CXX_STANDARD 11)

include_directories(./ncnn/include)
add_library(ncnn STATIC IMPORTED) 
set_property(TARGET ncnn PROPERTY IMPORTED_LOCATION ../ncnn/libncnn.a)
target_link_libraries(pfpld_ncnn ncnn)

