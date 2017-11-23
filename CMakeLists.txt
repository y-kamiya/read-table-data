cmake_minimum_required(VERSION 2.8)

project( readTableImage CXX )

find_package( OpenCV REQUIRED )

set(GCC_COVERAGE_LINK_FLAGS    "-llept -ltesseract")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}" )


include_directories( ${OpenCV_INCLUDE_DIRS} )

set( CMAKE_CXX_FLAGS "-Wall --std=c++1z" )

add_executable( readTableImage src/main.cpp )
target_link_libraries( readTableImage ${OpenCV_LIBS} )
