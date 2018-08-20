cmake_minimum_required (VERSION 2.8)

SET (SRC_LIST mask_rcnn.cpp)
include_directories ( "/env/caffe/include"
    "${PROJECT_SOURCE_DIR}/../../lib/nms"
    /usr/local/include
    /usr/include/python2.7
    /usr/local/cuda/include )

add_library(mask_rcnn SHARED ${SRC_LIST})