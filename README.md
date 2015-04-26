# image-processing-computer-vision
Repository containing the assignments done for the unit of Image Processing and Computer Vision at the University of Bristol in 2013/2014.

All programs were writen in C++ using the OpenCV library.

Assignment 1: coin counting
	A program to count and highlight coins in a image

Instalation guide:

This guide is intendend for Linux machines. If you wish to install OpenCV in other systems, check the other guides here: http://docs.opencv.org/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html#table-of-content-introduction

Try to adapt the installation instructions for your system with the ones bellow.


1- Install these packages:
	sudo apt-get install g++
	sudo apt-get install build-essential
	sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
	sudo apt-get install v4l2ucp v4l-utils libv4l-dev
	sudo apt-get install libgtk2.0-0 libgtk2.0-dev pkg-config

2 - Download this version of OpenCV: http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.11/

3 - Extract it to /usr/local/:
	
	unzip opencv-2.4.11.zip -d /usr/local/

4 - Rename opencv-2.4.11 to opencv-2.4 so the opencv folder becomes /usr/local/opencv-2.4.11/:
	mv /usr/local/opencv-2.4.11/ /usr/local/opencv-2.4

5 - Enter /usr/local/opencv-2.4/ and create a temporary directory where you want to put the generated MakeFiles and binaries, for instance:
	
	cd /usr/local/opencv-2.4.11/
	mkdir temp
	cd temp
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_GTK=ON -D WITH_V4L=ON -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-2.4/ ..
	make
	make install

6 - Include these lines in your .bashcr file (gedit ~/.bashrc):

	export LD_LIBRARY_PATH=/usr/local/opencv-2.4/lib:$LD_LIBRARY_PATH

	export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/local/opencv-2.4/include/opencv2/:/usr/local/opencv-2.4/include

	export OPENCV_CFLAGS=-I/usr/local/opencv-2.4/include/opencv2/

	export O_LIBS="-L/usr/local/opencv-2.4/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann"

7 - After editing your .bashrc file, make sure you refresh your environment via:

	source ~/.bashrc

