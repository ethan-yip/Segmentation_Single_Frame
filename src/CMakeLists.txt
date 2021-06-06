# 编译规则
add_compile_options(-std=c++14)

# 增加PCL库的依赖
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

# 增加opencv的依赖
FIND_PACKAGE( OpenCV REQUIRED )

# 添加头文件和库文件
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )


#添加一个头文件，生成一个库
ADD_LIBRARY( depth_segmentation depth_segmentation.cpp )
TARGET_LINK_LIBRARIES( depth_segmentation
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

#生成可执行文件
ADD_EXECUTABLE( singleSeg singleSeg.cpp )
TARGET_LINK_LIBRARIES( singleSeg 
    depth_segmentation
        common
	glog
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )
