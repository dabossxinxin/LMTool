
link_directories(${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(testCurveFitting CurveFitting.cpp)
target_link_libraries(testCurveFitting 
	${OpenCV_LIBRARIES}
	${PROJECT_NAME}_backend
	#${PROJECT_NAME}_utils
)

add_executable(testHessianNullSpace HessianNullSpace.cpp)
target_link_libraries(testHessianNullSpace
	${OpenCV_LIBRARIES}
	${PROJECT_NAME}_backend
	#${PROJECT_NAME}_utils
)