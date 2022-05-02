
link_directories(${POJECT_SOURCE_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

file(GLOB UtilsHdrs "./*.h")
file(GLOB UtilsSrcs "./*.cpp")

source_group("Source Files" FILES ${UtilsSrcs}) 
source_group("Header Files" FILES ${UtilsHdrs}) 

message("Utils Sources: " ${UtilsSrcs})
message("Utils Headers: " ${UtilsHdrs})

add_library(${PROJECT_NAME}_utils 
	${UtilsSrcs} 
	${UtilsHdrs})
	
target_link_libraries(${PROJECT_NAME}_utils
	${OpenCV_LIBRARIES})