// Image Processing and Computer Vision
// Assignment 03: Gesture Recognition
// Jia Zou - Joao Vitor Brandao Moreira - Matheus Mostiack Pomaleski

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void myNormalize(cv::Mat &input) {
    cv::normalize(input, input, 0, 255, 32, -1);
    input.convertTo(input, CV_8UC1);
}

#define SIZE 9 //Width and height of the region over which we compute the motion 
#define THRESH 1 //minimum size of the motion vectors for them to be considered motion (used to discard noise)
#define DELAY 1 //distance between the current frame and the frame we consider the previous on in our calculations - DELAY 1 means that we use one frame and the next one

/*computes Ix, Iy and It for the whole image
  we use the same idea of the slide 14 of the lecture 8*/
void gradient(Mat &pframe, Mat &frame, Mat &ix, Mat &iy, Mat &it){
	int x,y,w,h;
	x=1;y=1;w=ix.rows;h=ix.cols;
	for (int i=x; i<x+w; i++){
		for (int j=y;j<y+h; j++){
			ix.at<double>(i-1,j-1) = ((pframe.at<uchar>(i-1, j) - pframe.at<uchar>(i-1,j-1)) + (pframe.at<uchar>(i, j) - pframe.at<uchar>(i,j-1)) + (frame.at<uchar>(i-1, j) - frame.at<uchar>(i-1,j-1)) + (frame.at<uchar>(i, j) - frame.at<uchar>(i,j-1))) / 4;
			iy.at<double>(i-1,j-1) = ((pframe.at<uchar>(i, j-1) - pframe.at<uchar>(i-1,j-1)) + (pframe.at<uchar>(i, j) - pframe.at<uchar>(i-1,j)) + (frame.at<uchar>(i, j-1) - frame.at<uchar>(i-1,j-1)) + (frame.at<uchar>(i, j) - frame.at<uchar>(i-1,j))) / 4;
			it.at<double>(i-1,j-1) = ((frame.at<uchar>(i-1, j-1) - pframe.at<uchar>(i-1,j-1)) + (frame.at<uchar>(i-1, j) - pframe.at<uchar>(i-1,j)) + (frame.at<uchar>(i, j-1) - pframe.at<uchar>(i,j-1)) + (frame.at<uchar>(i, j) - pframe.at<uchar>(i,j))) / 4;
		//	printf("%lf\n", it.at<double>(i-1,j-1));
		}
	}
	return;
}

//executes the Lucas and Kanade algorithm
Mat lk(Mat &ix, Mat &iy, Mat &it, int j0, int j1, int i0, int i1){
double sumix2,sumiy2,ixiy,ixit,iyit;
	for (int i=i0;i<=i1;i++)
		for (int j=j0;j<=j1;j++){
			sumix2+=(ix.at<double>(i,j)*ix.at<double>(i,j));
			sumiy2+=(iy.at<double>(i,j)*iy.at<double>(i,j));
			ixiy+=(ix.at<double>(i,j)*iy.at<double>(i,j));
			ixit+=(ix.at<double>(i,j)*it.at<double>(i,j));
			iyit+=(iy.at<double>(i,j)*it.at<double>(i,j));
		}
	Mat A =Mat(2,2,CV_64FC1);
	Mat b =Mat(2,1,CV_64FC1);
	A.at<double>(0,0)=sumix2;
	A.at<double>(0,1)=ixiy;
	A.at<double>(1,0)=ixiy;
	A.at<double>(1,1)=sumiy2;
	b.at<double>(0,0)=-ixit;
	b.at<double>(0,1)=-iyit;
	//if the matrix doesn't have an inverse, we calculate a pseudo-inverse
	Mat v= A.inv(DECOMP_SVD)*b;
//	printf("vx is %lf, vy is %lf\n",v.at<double>(0,0), v.at<double>(0,1));
	return v;
}

//computes each of the regions that we want to calculate the lk function at, and calls the lk function for these regions
void lkTracker(Mat &image, Mat &ix, Mat &iy, Mat &it, Mat &vx, Mat &vy){
	int smallSize = SIZE/2; //the number of pixels between each border of the region and its center
	
	Point center;
	int i, j;
	Mat v;
	double xmin = 0, xmax = 0, ymin = 0, ymax = 0;
	//iterates through the image, calculating the motion vector and drawing it if it is bigger than the threshold 
	for(center.y = smallSize; center.y < ix.rows - SIZE; center.y += SIZE){
		for(center.x = smallSize; center.x < ix.cols - SIZE; center.x += SIZE){
			v = lk(ix,iy,it, center.x - smallSize, center.x+smallSize, center.y - smallSize, center.y + smallSize);
			if(v.at<double>(0, 0) > THRESH || v.at<double>(0, 0) < -THRESH || v.at<double>(0, 1) > THRESH || v.at<double>(0, 1) < -THRESH)
				line(image, center, Point((int)(center.x+v.at<double>(0,0)),(int)(center.y+v.at<double>(0,1))), Scalar(0, 0, 255), 1, 8, 0);
			vx.at<double>(center.y/SIZE, center.x/SIZE) = v.at<double>(0, 0);
			vy.at<double>(center.y/SIZE, center.x/SIZE) = v.at<double>(0, 1);
		}
	}
	

}

//returns 1 if there is movement to the right/bottom, -1 if there's movement to the top/left and 0 if there's no movement
int detectMovement(Mat &v){
	double sum = 0;
	int i, j;
	for(i = 0; i < v.rows; i++)
		for(j = 0; j < v.cols; j++)
			if(v.at<double>(i, j) < -THRESH || v.at<double>(i, j) > THRESH)
				sum += v.at<double>(i, j);
	if(sum > 30*THRESH)
		return 1;
	else if(sum < -30*THRESH)
		return -1;
	else return 0;
}

void normalizeV(Mat &v){
	int i, j;
	for(i = 0; i < v.rows; i++){
		for(j = 0; j < v.cols; j++){
				v.at<double>(i, j) = 0.5 + v.at<double>(i, j)/40;		
		}
	}
	return;
}

int main( int argc, const char** argv )
{
	cv::VideoCapture cap;
	if(argc > 1)
	{
		cap.open(string(argv[1]));
	}
	else
	{
		cap.open(CV_CAP_ANY);
	}
	if(!cap.isOpened())
	{
		printf("Error: could not load a camera or video.\n");
	}
	
	Mat frame[DELAY+1];
	Mat gray[DELAY+1];
	Mat ix, iy, it;
	Mat display;
	Mat vx, vy;

	int counter = 0;
	cap >> frame[0];
	
	//if we are receiving the input from the webcam, we resize the image to 1/3 of its size, because then we can compute the motion in real time
	if(argc == 1)
		resize(frame[0], frame[0], Size(0,0), 1/(double)3, 1/(double)3, INTER_LINEAR );
	//we convert all the input images to grayscale
	cvtColor( frame[0], gray[0], CV_BGR2GRAY );
	ix = Mat(frame[0].rows - 1, frame[0].cols - 1, CV_64FC1);
	iy = Mat(frame[0].rows - 1, frame[0].cols - 1, CV_64FC1);
	it = Mat(frame[0].rows - 1, frame[0].cols - 1, CV_64FC1);
	vx = Mat((frame[0].rows - 1)/SIZE, (frame[0].cols - 1)/SIZE, CV_64FC1);
	vy = Mat((frame[0].rows - 1)/SIZE, (frame[0].cols - 1)/SIZE, CV_64FC1);
	Mat vxDisplay;
	Mat vyDisplay;
	int i;
	for(i = 0; i < DELAY+1; i++){
		cap >> frame[DELAY-i];
		if(argc == 1)
			resize(frame[DELAY-i], frame[DELAY-i], Size(0,0), 1/(double)3, 1/(double)3, INTER_LINEAR );
		cvtColor( frame[DELAY-i], gray[DELAY-i], CV_BGR2GRAY );
		waitKey(20);
	}
	for(;;)
	{
		counter++;
		//discards the oldest frame and moves all the others, opening space to read a new frame
		for(i = DELAY; i > 0; i--){
			frame[i].release();
			gray[i].release();
			frame[i-1].copyTo(frame[i]);
			gray[i-1].copyTo(gray[i]);
		}
		
		waitKey(20);
		cap >> frame[0];
		if(argc == 1)
			resize(frame[0], frame[0], Size(0,0), 1/(double)3, 1/(double)3, INTER_LINEAR );
		cvtColor( frame[0], gray[0], CV_BGR2GRAY );
		
		if(!frame[0].data)
		{
			printf("Error: no frame data.\n");
			break;
		}
		

		display.release();
		frame[0].copyTo(display);

		//calculate the motion between the current frame and frame[DELAY]
		ix.convertTo(ix, CV_64FC1);
		iy.convertTo(iy, CV_64FC1);
		it.convertTo(it, CV_64FC1);
		
		gradient(gray[DELAY], gray[0], ix, iy, it);
		lkTracker(display, ix,iy,it,vx, vy);
		int hor = detectMovement(vx);
		int vert = detectMovement(vy);
		if(hor > 0)
			putText(display, "Right", Point(3*display.cols/5,display.rows/2), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 1, 8, false );
		else if(hor < 0)
			putText(display, "Left", Point(0,display.rows/2), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 1, 8, false );

		if(vert < 0)
			putText(display, "Up", Point(2*display.cols/5, display.rows/4), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 1, 8, false );
		else if(vert > 0)
			putText(display, "Down", Point(2*display.cols/5, 5*display.rows/6), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 1, 8, false );

		//we normalize the vectors in order to display them
		normalizeV(vx);
		normalizeV(vy);
		//we also expand them so that they have a similar size to the input image
		resize(vx, vxDisplay, Size(0,0), SIZE, SIZE, INTER_NEAREST );
		resize(vy, vyDisplay, Size(0,0), SIZE, SIZE, INTER_NEAREST );
		myNormalize(ix);
		myNormalize(iy);
		myNormalize(it);
		imshow("ix", ix);
	 	imshow("iy", iy);
	 	imshow("it", it);
		imshow("vx", vxDisplay);
		imshow("vy", vyDisplay);
		imshow("pframe", frame[DELAY]);
		imshow("video", display);
			
	}
}



