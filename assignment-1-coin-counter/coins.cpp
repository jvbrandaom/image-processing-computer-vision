// Image Processing and Computer Vision
// Assignment 01: Coin Counter
// Jia Zou - Joao Vitor Brandao Moreira - Matheus Mostiack Pomaleski

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <opencv.hpp>
#include <highgui/highgui.hpp>
 
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
    //cv::normalize(aux, aux, 0, 255, 32, -1);
    //aux.convertTo(output, CV_8UC1);
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
    int i, j;
    int sizes[3] = {mag.rows, mag.cols, mag.rows/2};
    houghSpace = cv::Mat(3, sizes, CV_64FC1, double(0));
    //printf("here");
    //houghSpace.at<double>(mag.rows-1, mag.cols-1, mag.rows/2-1) = 123;
    
    //printf("%f\n", houghSpace.at<double>(mag.rows-1, mag.cols-1, mag.rows/2-1));
  for (int i = 0; i < mag.rows; i++)
      for (int j = 0; j < mag.cols; j++)
      	if (mag.at<uchar> (i, j) == 255) {
          for (int r = 20; r < maxRadius; r++) {
           
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
        
/*  for (int x0 = 0; x0 < mag.rows; x0++)
      for (int y0 = 0; y0 < mag.cols; y0++)
          for (int r = 0; r < maxRadius; r++) {
              if (houghSpace.at<double>(x0, y0, r) > 20) {
                  printf("found circle at %d %d, radius = %d, %lf votes\n", x0, y0, r, houghSpace.at<double>(x0, y0, r));
                  for(int a = -10; a < 10; a++){
                  	mag.at<uchar>(x0+a,y0) = 128;
                  	mag.at<uchar>(x0,y0+a) = 128;
		  }
              }
          }
  */  return;
}
 
void displayHough(cv::Mat &hough, cv::Mat &output, int nrows, int ncols){
	output = cv::Mat(nrows, ncols, CV_64FC1, double(0));
	for (int i = 0; i < nrows; i++)
      	  for (int j = 0; j < ncols; j++)
            for (int r = 10; r < nrows/2; r++)
            	output.at<double>(i, j) += hough.at<double>(i,j,r);
             
	
	myNormalize(output);
	return;
}

int isBestCenter(cv::Mat &houghSpace, int x0, int y0, int r, int nrows, int ncols){
	for(int i = 0; i <= r/2; i++){
		for(int j = 0; j <= r/2; j++){
			if(i==0 && j==0)
				continue;
			for(int k = 10; k < nrows/2; k++){
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

void circles(cv::Mat &houghSpace, cv::Mat &image, int threshold){
   int ncircles = 0;
   for (int x0 = 0; x0 < image.rows; x0++)
      for (int y0 = 0; y0 < image.cols; y0++)
          for (int r = 20; r < image.rows/2; r++) {
              if (houghSpace.at<double>(x0, y0, r) > threshold && isBestCenter(houghSpace, x0, y0, r, image.rows, image.cols)) {
		  ncircles++;
//                  printf("found circle at %d %d, radius = %d, %lf votes\n", x0, y0, r, houghSpace.at<double>(x0, y0, r));
		  circle(image, cv::Point(y0,x0), r-1, 255, 1, 8, 0);
          circle(image, cv::Point(y0,x0), r, 0, 1, 8, 0);
          circle(image, cv::Point(y0,x0), r+1, 255, 1, 8, 0);
          circle(image, cv::Point(y0,x0), r+2, 0, 1, 8, 0);

                  /*for(int a = -10; a < 10; a++){
                  	mag.at<uchar>(x0+a,y0) = 128;
                  	mag.at<uchar>(x0,y0+a) = 128;
		  }*/
              }
          }
          printf("%d circles were found\n", ncircles);
          return;
}	

int main(int argc, char ** argv) {
//Image 1
    cv::Mat coins1 = cv::imread("images/coins1.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat ddx1;
    cv::Mat ddy1;
    cv::Mat magnitude1;
    cv::Mat gradient1;
    cv::Mat magThreshold1;
    cv::Mat houghSpace1;
    cv::Mat houghSpaceDisplay1;
    
//    ddx(coins1, ddx1);
//    ddy(coins1, ddy1);
    magnitude(coins1, magnitude1);
    gradientDirection(coins1, gradient1);
    magThreshold(magnitude1, magThreshold1, 32);
    hough(magThreshold1, gradient1, houghSpace1);
//    displayHough(houghSpace1, houghSpaceDisplay1, magThreshold1.rows, magThreshold1.cols);
    circles(houghSpace1, coins1, 21); 
 
/*    myNormalize(ddx1);
    myNormalize(ddy1);
    myNormalize(magnitude1);
    myNormalize(gradient1);
 
    cv::namedWindow("Ddx1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Ddy1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Magnitude1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Gradient Direction1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Magnitude threshold1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Hough Space1", CV_WINDOW_AUTOSIZE);
*/   cv::namedWindow("Circled1", CV_WINDOW_AUTOSIZE);
 
//    cv::imshow("Ddx1", ddx1);
//    cv::imshow("Ddy1", ddy1);
//    cv::imshow("Magnitude1", magnitude1);
//    cv::imshow("Gradient Direction1", gradient1);
//    cv::imshow("Magnitude threshold1", magThreshold1);
//    cv::imshow("Hough Space1", houghSpaceDisplay1);
    cv::imshow("Circled1", coins1);

//Image 2
    cv::Mat coins2 = cv::imread("images/coins2.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat ddx2;
    cv::Mat ddy2;
    cv::Mat magnitude2;
    cv::Mat gradient2;
    cv::Mat magThreshold2;
    cv::Mat houghSpace2;
    cv::Mat houghSpaceDisplay2;
    
//    ddx(coins2, ddx2);
//    ddy(coins2, ddy2);
    magnitude(coins2, magnitude2);
    gradientDirection(coins2, gradient2);
    magThreshold(magnitude2, magThreshold2, 32);
    hough(magThreshold2, gradient2, houghSpace2);
//    displayHough(houghSpace2, houghSpaceDisplay2, magThreshold2.rows, magThreshold2.cols);
    circles(houghSpace2, coins2, 22); 
 
/*    myNormalize(ddx2);
    myNormalize(ddy2);
    myNormalize(magnitude2);
    myNormalize(gradient2);
 
    cv::namedWindow("Ddx2", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Ddy2", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Magnitude2", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Gradient Direction2", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Magnitude threshold2", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Hough Space2", CV_WINDOW_AUTOSIZE);
*/   cv::namedWindow("Circled2", CV_WINDOW_AUTOSIZE);
 
//    cv::imshow("Ddx2", ddx2);
//    cv::imshow("Ddy2", ddy2);
//    cv::imshow("Magnitude2", magnitude2);
//    cv::imshow("Gradient Direction2", gradient2);
//    cv::imshow("Magnitude threshold2", magThreshold2);
//    cv::imshow("Hough Space2", houghSpaceDisplay2);
    cv::imshow("Circled2", coins2);

//Image 3
    cv::Mat coins3 = cv::imread("images/coins3.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat ddx3;
    cv::Mat ddy3;
    cv::Mat magnitude3;
    cv::Mat gradient3;
    cv::Mat magThreshold3;
    cv::Mat houghSpace3;
    cv::Mat houghSpaceDisplay3;
    
//    ddx(coins3, ddx3);
//   ddy(coins3, ddy3);
    magnitude(coins3, magnitude3);
    gradientDirection(coins3, gradient3);
    magThreshold(magnitude3, magThreshold3, 20);
    hough(magThreshold3, gradient3,houghSpace3);
//    displayHough(houghSpace3, houghSpaceDisplay3, magThreshold3.rows, magThreshold3.cols);
    circles(houghSpace3, coins3, 12); 
 
//    myNormalize(ddx3);
//    myNormalize(ddy3);
    myNormalize(magnitude3);
    myNormalize(gradient3);
 
//    cv::namedWindow("Ddx3", CV_WINDOW_AUTOSIZE);
//    cv::namedWindow("Ddy3", CV_WINDOW_AUTOSIZE);
//    cv::namedWindow("Magnitude3", CV_WINDOW_AUTOSIZE);
//    cv::namedWindow("Gradient Direction3", CV_WINDOW_AUTOSIZE);
//    cv::namedWindow("Magnitude threshold3", CV_WINDOW_AUTOSIZE);
//    cv::namedWindow("Hough Space3", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Circled3", CV_WINDOW_AUTOSIZE);
 
//    cv::imshow("Ddx3", ddx3);
//    cv::imshow("Ddy3", ddy3);
//    cv::imshow("Magnitude3", magnitude3);
//    cv::imshow("Gradient Direction3", gradient3);
//    cv::imshow("Magnitude threshold3", magThreshold3);
//    cv::imshow("Hough Space3", houghSpaceDisplay3);
    cv::imshow("Circled3", coins3);
 
    
    cv::waitKey();
 
    return 0;
}

