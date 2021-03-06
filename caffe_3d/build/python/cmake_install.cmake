# Install script for directory: /home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE FILE FILES
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/polyak_average.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/classify.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/convert_to_fully_conv.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/gen_bn_inference.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/bn_convert_style.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/draw_net.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/detect.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/requirements.txt"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE FILE FILES
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/caffe/pycaffe.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/caffe/classifier.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/caffe/detector.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/caffe/__init__.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/caffe/net_spec.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/caffe/io.py"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/caffe/draw.py"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so"
         RPATH "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/build/install/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/usr/local/cuda-8.0/lib64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE SHARED_LIBRARY FILES "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/build/lib/_caffe.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so"
         OLD_RPATH "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/build/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/usr/local/cuda-8.0/lib64::::::::"
         NEW_RPATH "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/build/install/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/usr/local/cuda-8.0/lib64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE DIRECTORY FILES
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/caffe/imagenet"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/caffe/proto"
    "/home/kora/Downloads/ECO-efficient-video-understanding-master/caffe_3d/python/caffe/test"
    )
endif()

