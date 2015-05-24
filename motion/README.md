Assingment 3: Gesture Recognition

This program is going to take your webcam input and indicate the general direction of motion. It opens 7 windows showing the intermediate steps (gradient images, derivatives and so on), the one named "video" being the one in which the motion detection takes place.

We created a rudimentary gesture recognition system based on the Lucas-Kanade
(LK) optical flow method. In order to do that, we firstly computed the image derivatives Ix, Iy and It. Then
we implemented the LK algorithm, that divides the frames into regions, and calculates the motion vectors
for each of these regions. Lastly, we defined a threshold to distinguish noise from actual movement, and
summed the horizontal and vertical components from the motion vectors that represent real movement.
With this, we were able to detect gestures in both the horizontal as the vertical axes.

Known issues:

- The gesture recognition depends on the resolution of your webcam. The program was written to run well in intermediate resolution webcams. Using HD webcams, the program will detect motion only if it is very close to the camera.
- The program may "crash" your webcam, which means you can run the program sucessfully in the first atempt but your webcam is not going to work afterwards unless you restart your computer (this happened specifically on my computer)

Compiling and running instructions:

g++ video.cpp /usr/local/opencv-2.4/lib/libopencv_core.so.2.4 /usr/local/opencv-2.4/lib/libopencv_highgui.so.2.4 /usr/local/opencv-2.4/lib/libopencv_imgproc.so.2.4

./a.out

