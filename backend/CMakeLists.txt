
link_directories(${POJECT_SOURCE_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

file(GLOB BackEndHdrs "./*.h")
file(GLOB BackEndSrcs "./*.cpp")

source_group("Source Files" FILES ${BackEndSrcs})
source_group("Header Files" FILES ${BackEndHdrs})

add_library(${PROJECT_NAME}_backend
	${BackEndHdrs}
	${BackEndSrcs})

target_link_libraries(${PROJECT_NAME}_backend
	${OpenCV_LIBRARIES})