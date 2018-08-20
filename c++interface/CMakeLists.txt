#This part is used for compile mask_rcnn_demo.cpp
cmake_minimum_required (VERSION 2.8)

project (main_demo)

add_executable(main main.cpp)

include_directories ( "/env/caffe/include"
    "${PROJECT_SOURCE_DIR}/../lib/nms"
    "${PROJECT_SOURCE_DIR}/lib"
    /usr/local/include
    /usr/include/python2.7
    /usr/local/cuda/include )

target_link_libraries(main ${PROJECT_SOURCE_DIR}/lib/libmask_rcnn.so
    /env/caffe/build/lib/libcaffe.so
    ${PROJECT_SOURCE_DIR}/../lib/nms/gpu_nms.so
    /usr/lib/x86_64-linux-gnu/libopencv_highgui.so
    /usr/lib/x86_64-linux-gnu/libopencv_core.so
    /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so
    /usr/lib/x86_64-linux-gnu/libglog.so
    /usr/lib/x86_64-linux-gnu/libboost_system.so
    /usr/lib/x86_64-linux-gnu/libboost_python.so
    /usr/lib/x86_64-linux-gnu/libglog.so
    /usr/lib/x86_64-linux-gnu/libpython2.7.so
    )