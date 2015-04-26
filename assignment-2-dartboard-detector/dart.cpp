// Image Processing and Computer Vision
// Assignment 02: Object Recognition
// Jia Zou - Joao Vitor Brandao Moreira - Matheus Mostiack Pomaleski


#include "opencv.hpp"
#include "objdetect/objdetect.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
 
#include <iostream>
#include <stdio.h>
 
using namespace std;
using namespace cv;

void ddy(cv::Mat &input, cv::Mat &aux) {
    int size = 3;
    // initialise the output using the input
    // initialise kernel
    double kernel[size][size];
    kernel[0][0] = -1;
    kernel[0][1] = -2;
    kernel[0][2] = -1;
    kernel[1][0] = 0;
    kernel[1][1] = 0;
    kernel[1][2] = 0;
    kernel[2][0] = 1;
    kernel[2][1] = 2;
    kernel[2][2] = 1;
    //put kernel into Mat form
    cv::Mat Kernel = cv::Mat(size, size, CV_64FC1, kernel);
    aux = cv::Mat(input.rows, input.cols, CV_64FC1);

    int kernelRadiusX = (Kernel.size[0] - 1) / 2;
    int kernelRadiusY = (Kernel.size[1] - 1) / 2;

    cv::Mat paddedInput;
    cv::copyMakeBorder(input, paddedInput, kernelRadiusX, kernelRadiusX,
            kernelRadiusY, kernelRadiusY, cv::BORDER_REPLICATE);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            double sum = 0.0;
            for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
                for (int n = -kernelRadiusY; n <= kernelRadiusY; n++) {
                    // find the correct indices we are using
                    int imagex = i + 1 + m;
                    int imagey = j + 1 + n;
                    int kernelx = m + kernelRadiusX;
                    int kernely = n + kernelRadiusY;

                    // get the values from the padded image and the kernel
                    int imageval = (int) paddedInput.at<uchar> (imagex, imagey);
                    double kernalval = Kernel.at<double> (kernelx, kernely);

                    // do the multiplication
                    sum += imageval * kernalval;
                }
            }
            //save every pixel value into Mat aux
            aux.at<double> (i, j) = sum;
        }
    }
    //cv::normalize(aux, aux, 0, 255, 32, -1);
    //aux.convertTo(output, CV_8UC1);
}

void ddx(cv::Mat &input, cv::Mat &aux) {
    int size = 3;
    // initialise the output using the input
    // initialise kernel
    double kernel[size][size];
    kernel[0][0] = -1;
    kernel[0][1] = 0;
    kernel[0][2] = 1;
    kernel[1][0] = -2;
    kernel[1][1] = 0;
    kernel[1][2] = 2;
    kernel[2][0] = -1;
    kernel[2][1] = 0;
    kernel[2][2] = 1;
    //put kernel into Mat form
    cv::Mat Kernel = cv::Mat(size, size, CV_64FC1, kernel);
    aux = cv::Mat(input.rows, input.cols, CV_64FC1);

    int kernelRadiusX = (Kernel.size[0] - 1) / 2;
    int kernelRadiusY = (Kernel.size[1] - 1) / 2;

    cv::Mat paddedInput;
    cv::copyMakeBorder(input, paddedInput, kernelRadiusX, kernelRadiusX,
            kernelRadiusY, kernelRadiusY, cv::BORDER_REPLICATE);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            double sum = 0.0;
            for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
                for (int n = -kernelRadiusY; n <= kernelRadiusY; n++) {
                    // find the correct indices we are using
                    int imagex = i + 1 + m;
                    int imagey = j + 1 + n;
                    int kernelx = m + kernelRadiusX;
                    int kernely = n + kernelRadiusY;

                    // get the values from the padded image and the kernel
                    int imageval = (int) paddedInput.at<uchar> (imagex, imagey);
                    double kernalval = Kernel.at<double> (kernelx, kernely);

                    // do the multiplication
                    sum += imageval * kernalval;
                }
            }
            //save every pixel value into Mat aux
            aux.at<double> (i, j) = sum;
        }
    }
}

void magnitude(cv::Mat &input, cv::Mat &aux) {
    cv::Mat ddxImage, ddyImage;
    ddx(input, ddxImage);
    ddy(input, ddyImage);
    aux = cv::Mat(input.rows, input.cols, CV_64FC1);
    for (int i = 0; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++) {
            aux.at<double> (i, j) = sqrt(
                    pow(ddxImage.at<double> (i, j), 2) + pow(
                            ddyImage.at<double> (i, j), 2));
        }
}

void gradientDirection(cv::Mat &input, cv::Mat &aux) {
    cv::Mat ddxImage, ddyImage;
    ddx(input, ddxImage);
    ddy(input, ddyImage);
    aux = cv::Mat(input.rows, input.cols, CV_64FC1);
    for (int i = 0; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++) {
            aux.at<double> (i, j) = atan(
                    ddyImage.at<double> (i, j) / ddxImage.at<double> (i, j));
        }
}

void myNormalize(cv::Mat &input) {
    cv::normalize(input, input, 0, 255, 32, -1);
    input.convertTo(input, CV_8UC1);
}

void magThreshold(cv::Mat &input, cv::Mat &aux, int threshold) {
    cv::normalize(input, aux, 0, 255, 32, -1);
    aux.convertTo(aux, CV_8UC1);
    for (int i = 0; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++) {
            if (aux.at<uchar> (i, j) > threshold) {
                aux.at<uchar> (i, j) = 255;
            } else
                aux.at<uchar> (i, j) = 0;
        }

}
void hough(cv::Mat &mag, cv::Mat &gradient, cv::Mat &houghSpace) {
    int maxRadius = (mag.rows) / 2;
    int x0positive;
    int x0negative;
    int y0positive;
    int y0negative;
    int sizes[3] = {mag.rows, mag.cols, mag.rows/2};
    houghSpace = cv::Mat(3, sizes, CV_64FC1, double(0));
    //houghSpace.at<double>(mag.rows-1, mag.cols-1, mag.rows/2-1) = 123;
    //printf("%f\n", houghSpace.at<double>(mag.rows-1, mag.cols-1, mag.rows/2-1));
  for (int i = 0; i < mag.rows; i++)
      for (int j = 0; j < mag.cols; j++)
      	if (mag.at<uchar> (i, j) == 255) {
          for (int r = 25; r < maxRadius; r++) {
                   y0positive = j + r * cos(gradient.at<double> (i, j));
                   y0negative = j - r * cos(gradient.at<double> (i, j));
                   x0positive = i + r * sin(gradient.at<double> (i, j));
                   x0negative = i - r * sin(gradient.at<double> (i, j));
                  if (x0positive >= 0 && y0positive >= 0 && x0positive < mag.rows && y0positive < mag.cols) {
                      houghSpace.at<double>(x0positive, y0positive,r) = houghSpace.at<double>(x0positive, y0positive,r)+1;
                  }
                  if (x0negative >= 0 && y0negative >= 0 && x0negative < mag.rows && y0negative < mag.cols) {
                      houghSpace.at<double>(x0negative, y0negative, r) = houghSpace.at<double>(x0negative, y0negative, r)+ 1;
                  }
            }
          }
  return;
}

void displayHough(cv::Mat &hough, cv::Mat &output, int nrows, int ncols){
	output = cv::Mat(nrows, ncols, CV_64FC1, double(0));
	for (int i = 0; i < nrows; i++)
      	  for (int j = 0; j < ncols; j++)
            for (int r = 25; r < nrows/2; r++)
            	output.at<double>(i, j) += hough.at<double>(i,j,r);
	myNormalize(output);
	return;
}

int isBestCenter(cv::Mat &houghSpace, int x0, int y0, int r, int nrows, int ncols){
	int searchRange=r/2;
	for(int i = 0; i <= searchRange; i++){
		for(int j = 0; j <= searchRange; j++){
			if(i==0 && j==0)
				continue;
			for(int k = 20; k < nrows/2; k++){
				if(x0-i>=0 && y0-j>=0 && houghSpace.at<double>(x0-i, y0-j, k) >= houghSpace.at<double>(x0, y0, r)){
					if(houghSpace.at<double>(x0-i, y0-j, k) == houghSpace.at<double>(x0, y0, r))
						houghSpace.at<double>(x0, y0, r) = houghSpace.at<double>(x0, y0, r) - 1;
					return 0;
				}
				if(x0+i<nrows && y0-j>=0 && houghSpace.at<double>(x0+i, y0-j, k) >= houghSpace.at<double>(x0, y0, r)){
					if(houghSpace.at<double>(x0+i, y0-j, k) == houghSpace.at<double>(x0, y0, r))
						houghSpace.at<double>(x0, y0, r) = houghSpace.at<double>(x0, y0, r) - 1;
					return 0;
				}
				if(x0-i>=0 && y0+j<ncols && houghSpace.at<double>(x0-i, y0+j, k) >= houghSpace.at<double>(x0, y0, r)){
					if(houghSpace.at<double>(x0-i, y0+j, k) == houghSpace.at<double>(x0, y0, r))
						houghSpace.at<double>(x0, y0, r) = houghSpace.at<double>(x0, y0, r) - 1;
					return 0;
				}
				if(x0+i<nrows && y0+j<ncols && houghSpace.at<double>(x0+i, y0+j, k) >= houghSpace.at<double>(x0, y0, r)){
					if(houghSpace.at<double>(x0+i, y0+j, k) == houghSpace.at<double>(x0, y0, r))
						houghSpace.at<double>(x0, y0, r) = houghSpace.at<double>(x0, y0, r) - 1;
					return 0;
				}
			}
		}
	}
	return 1;
}

vector<Point> circles(cv::Mat &houghSpace, cv::Mat &image, int threshold){
   int ncircles = 0;
   vector<cv::Point> ret;
   for (int x0 = 0; x0 < image.rows; x0++)
      for (int y0 = 0; y0 < image.cols; y0++)
          for (int r = 20; r < image.rows/2; r++) {
              if (houghSpace.at<double>(x0, y0, r) > threshold && isBestCenter(houghSpace, x0, y0, r, image.rows, image.cols)) {
		  ncircles++;
//                  printf("found circle at %d %d, radius = %d, %lf votes\n", x0, y0, r, houghSpace.at<double>(x0, y0, r));
//		  circle(image, cv::Point(y0,x0), r, cv::Scalar(0, 0, 255), 2, 8, 0);
				ret.push_back(cv::Point(y0,x0));
              }
          }
//          printf("%d circles were found\n", ncircles);
          return ret;
}

vector<Point> findCircles(char fileName[]) {
//Image 1
    cv::Mat coins1original = cv::imread(fileName);
    cv::Mat coins1 = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat ddx1;
    cv::Mat ddy1;
    cv::Mat magnitude1;
    cv::Mat gradient1;
    cv::Mat magThreshold1;
    cv::Mat houghSpace1;
    cv::Mat houghSpaceDisplay1;

    ddx(coins1, ddx1);
    ddy(coins1, ddy1);
    magnitude(coins1, magnitude1);
    gradientDirection(coins1, gradient1);
    magThreshold(magnitude1, magThreshold1, 40);
    hough(magThreshold1, gradient1, houghSpace1);
    displayHough(houghSpace1, houghSpaceDisplay1, magThreshold1.rows, magThreshold1.cols);
    vector<cv::Point> ret = circles(houghSpace1, coins1original, 10);
/*
    myNormalize(ddx1);
    myNormalize(ddy1);
    myNormalize(magnitude1);
    myNormalize(gradient1);

    cv::namedWindow("Ddx1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Ddy1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Magnitude1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Gradient Direction1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Magnitude threshold1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Hough Space1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Circled1", CV_WINDOW_AUTOSIZE);

    cv::imshow("Ddx1", ddx1);
    cv::imshow("Ddy1", ddy1);
    cv::imshow("Magnitude1", magnitude1);
    cv::imshow("Gradient Direction1", gradient1);
    cv::imshow("Magnitude threshold1", magThreshold1);
    cv::imshow("Hough Space1", houghSpaceDisplay1);
    cv::imshow("Circled1", coins1original);
*/


//    cv::waitKey();

    return ret;
}
 
/** Function Headers */
void detectAndSave( Mat frame, vector<Vec4i> lines, char *c, vector<cv::Point> circleCenters);
 
/** Global variables */
String logo_cascade_name = "dartcascade.xml";
 
CascadeClassifier logo_cascade;
 
string window_name = "Capture - Face detection";
 
 

/** @function main */
int main( int argc, const char** argv )
{
 char prefix[5]= "dart";
char extension[5]=".jpg";
char number[3];
char fileName[20];
char outputFileName[20];
/*loops for every image*/
for (int fileNum=0; fileNum<12; fileNum++){
	sprintf(number,"%d",fileNum);
	strcpy(fileName,prefix);
	strcat(fileName,number);	
	strcat(fileName,extension);
	strcpy(outputFileName,"output");
	strcat(outputFileName,fileName);
 	printf("%s\n",outputFileName);

	/*runs the circle detection algorithm from last assignment
	  the function returns a vector with the centers of every circle*/
 	printf("Finding circles...\n");
	vector<cv::Point> circleCenters = findCircles(fileName);
	
	/*uses the opencv Probabilistic Hough Line Transform to detect the lines in the image*/
	printf("Finding lines...\n");
	Mat frame =imread( fileName, CV_LOAD_IMAGE_COLOR);
	Mat grey;
	cvtColor( frame, grey, CV_BGR2GRAY );
 
    cv::Mat magnitude1;
    cv::Mat magThreshold1;
     
    CvCapture* capture;

    //-- 1. Load the cascades
    //logo_cascade.load(logo_cascade_name);
    if( !logo_cascade.load( logo_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
     
    magnitude(grey, magnitude1);
    magThreshold(magnitude1, magThreshold1, 30);
    
   	cv::Mat cdst;
    cv::cvtColor(magThreshold1, cdst, CV_GRAY2BGR);
    cv::vector<cv::Vec4i> lines;
	/*after running the function, lines is a vector that has the starting and ending coordinates for each line*/
    cv::HoughLinesP(magThreshold1, lines, 1, CV_PI/180, 110, 50,10 );
 /*   for( size_t i = 0; i < lines.size(); i++ )
  {
    cv::Vec4i l = lines[i];
    line( cdst, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 1, CV_AA);
 
  }
    cv::namedWindow("detected lines", CV_WINDOW_AUTOSIZE);
    cv::imshow("detected lines", cdst);
*/     
     
    detectAndSave( frame, lines, outputFileName, circleCenters);
     
//   cv::waitKey();
     }
    return 0;
}

/*this function will check if there are any possible dartboards that overlap with the current dartboard*/
int isBestSquare(std::vector<Rect> faces, int i, int nCenters[]){
	int y0, y1, x0, x1;
	y0 = faces[i].y;
	y1 = faces[i].y + faces[i].height;
	x0 = faces[i].x;
	x1 = faces[i].x + faces[i].width;
	for(int j = 0; j <= faces.size(); j++){
		if(i == j || nCenters[j] == 0)
			continue;
		if(((faces[j].x >= x0 && faces[j].x <= x1) || (faces[j].x + faces[j].width >= x0 && faces[j].x + faces[j].width <= x1)) && ((faces[j].y >= y0 && faces[j].y <= y1) || (faces[j].y + faces[j].height >= y0 && faces[j].y + faces[j].height <= y1))){
			/*when we find overlapping dartboards, we use the number of circles each possible dartboard 
			has divided by the possible dartboard area to decide which is the most likely real dartboard
			the other possible dartboard is then discarded*/
			if((float)nCenters[i]/(float)(faces[i].height*faces[i].width) >= (float)nCenters[j]/(float)(faces[j].height*faces[j].width))
				nCenters[j] = 0;
			else{
				nCenters[i] = 0;
				return 0;
			}
		}
	}
	return 1;
} 
  
/** @function detectAndSave */
void detectAndSave( Mat frame, vector<Vec4i> lines, char *c, vector<cv::Point> circleCenters)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    int averageX, averageY, lineSize;
    printf("Detecting dartboards...\n");
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
 
    /*firstly we use the detctMultiScaleFunction to detect the possible dartboards*/
    /*we noticed that it generates a lot of false positives, and no false negatives*/
    
    logo_cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  	/*so now we use the detected lines and circles to reduce the number of false positives*/
    int numberLines[faces.size()];
	int nCenters[faces.size()];
    for( int i = 0; i < faces.size(); i++ )
    {
        numberLines[i] = 0;
        /*firstly we assume that real dartboards have lots of lines inside them*/
        /*so we remove the possible dartboards that dont have a minimum number of lines*/
        for(int j = 0; j < lines.size(); j++)
        {
            averageX = (lines[j][0] + lines[j][2])/2;
            averageY = (lines[j][1] + lines[j][3])/2;
            if(averageX > faces[i].x && averageX < faces[i].x + faces[i].width && averageY > faces[i].y && averageY < faces[i].y + faces[i].height)
                numberLines[i] += 1;
        }
		nCenters[i] = 0;
        if(numberLines[i] > 18){
		/*we also assume that real dartboards have circles, that have centers that are inside them*/
		/*so we count the number of circles that each possible dartboard have inside them*/
        	for(int k = 0; k < circleCenters.size(); k++){
        		if(circleCenters[k].x >= faces[i].x && circleCenters[k].x <= faces[i].x + faces[i].width && circleCenters[k].y >= faces[i].y && circleCenters[k].y <= faces[i].y + faces[i].height)
        			nCenters[i]++;
        	}
        }
    }
    for( int i = 0; i < faces.size(); i++ ){
    	/*we will only consider dartboards that have at least 1 circle inside them*/
  		if(nCenters[i] > 0)
  			/*we assume that there are no overlapping dartboards, so we use the isBestSquare function to 
  			test if there are any other possible dartboards that are more likely to be the real dartboards*/
    		if(isBestSquare(faces, i, nCenters))
            	rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
/*//we can print the center of each detected circle into the resulting image    
    for (int ii = 0; ii < circleCenters.size(); ii++){
 		circle(frame, circleCenters[ii], 2, cv::Scalar(0, 0, 255), 2, 8, 0);
 	}
*/  //-- Save what you got
	imwrite( c, frame );
 
}
