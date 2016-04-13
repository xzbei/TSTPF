#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "defs.h"
#include "particles.h"
#include "observation.h"
#include "time.h"
#include "unistd.h"
#include "opencv/cv.h"
#include "TST.hpp"
#include "TrackMTD.hpp"
#include "BTD.hpp"
#include "globaldata.hpp"
#include "TSTRecognizer.hpp"
#include "PFTrack.hpp"

using namespace std;
using namespace cv;
using namespace GlobalConst;
using namespace GlobalVar;

#define MODE_RESET -1
#define MODE_BEGIN 0
#define MODE_TRAIN 1
#define MODE_TEST 2

CvScalar color;
char* vid_file = "/Users/beixinzhu/Documents/UCLA/visionlab/testmovie/bag.mp4";
void on_mouse(int event, int x, int y, int flags, void* param );
int is_detect = 0;

int main( int argc, char** argv )
{
  CvCapture* video;
  video = cvCaptureFromFile( vid_file );
    //video = cvCaptureFromCAM( 0);
    if( ! video ){
        printf("couldn't open video file %s", vid_file);
//        return 1;
    }

  mode = MODE_BEGIN;

  while( frame = cvQueryFrame( video ) ){
    IplImage* framegrey = cvCreateImage(cvGetSize(frame), 8, 1);
    cvCvtColor(frame, framegrey, CV_BGR2GRAY);
    width = frame->width;
    height = frame->height;
    IplImage* hsv_frame;
    hsv_frame = bgr2hsv(frame);

    switch (mode){
        case MODE_BEGIN:
            printf("MODE_BEGIN, numframes = %d\n",numframes);
            if (numframes == -1){
                initializeTracker(frame,framegrey);
                mode =  MODE_RESET;
                break;
            }
            TST_prep(frame,framegrey);
            TST_BEGIN(frame,framegrey);
            // rectangle(mGr,Point((ycenter-ycenter/2)*SCALE,(xcenter - xcenter/2)*SCALE), Point((ycenter+ycenter/2)*SCALE,(xcenter+xcenter/2)*SCALE),Scalar(255,0,0,255),8);
            mode =  MODE_BEGIN;
            break;
        case MODE_RESET:
            printf("MODE_RESET, numframes = %d\n",numframes);
            numframes = 0;
            TSTfirst_test = true;
            TST_prep(frame,framegrey);
            TST_RESET(frame,framegrey);
            // PF_RESET();
            mode =  MODE_BEGIN;
            break;
        case MODE_TRAIN:
            printf("MODE_TRAIN, numframes = %d\n",numframes);
            numframes ++;
            TST_prep(frame, framegrey);
            TST_TRAIN(frame, framegrey);
            PF_train(frame,hsv_frame);
            mode =  MODE_TEST;
            break;
        case MODE_TEST:
            printf("MODE_TEST, numframes = %d\n",numframes);
            cvWaitKey(0);
            numframes ++;
            TST_prep(frame, framegrey);
            is_detect = TST_TEST(frame, framegrey);
//            num_match = 1;

            if (TSTfirst_test && is_detect){
              PF_init(frame,hsv_frame);
              PF_test(frame, hsv_frame , TSTfirst_test);
              TSTfirst_test = false;
            }else{
              PF_test(frame, hsv_frame , TSTfirst_test);
            }
            
            mode =  MODE_TEST;
            break;
    }

    printf("numframes = %d\n", numframes);
    cvNamedWindow( "Video", 1 );

    cvShowImage( "Video", frame );
    cvSetMouseCallback("Video", &on_mouse,0);
//    cvWaitKey(0);

    if(cvWaitKey( 5 ) == 27)
        break;
//    cvDestroyAllWindows();
//    cvReleaseCapture( &video );

  }
  return 0;
}

void on_mouse(int event, int x, int y, int flags, void* param )
{
  if (mouseclick == false){
    // mouseclick = true;
    switch (event) {
      case CV_EVENT_LBUTTONDOWN:
        if (mode == MODE_BEGIN){
            mouseclick = true;
          mode = MODE_TRAIN;
            printf("System start training...\n");
            break;
        }
//        if (mode == MODE_TEST){
//          mode = MODE_RESET;
//          numframes = 0;
//            printf("System start resetting...\n");
//          initializeTracker(frame,framegrey);
//          TST_prep(frame,framegrey);
//          TST_RESET(frame,framegrey);
//
//            break;
//        }
            break;
//      case CV_EVENT_LBUTTONUP:
//        mouseclick = true;
//        if (mode == MODE_TRAIN){
//            printf("System start testing...\n");
//          mode = MODE_TEST;
//
//            break;cvWaitKey
//        }
//        if (mode == MODE_RESET){
//            printf("System start beginning...\n");
//          mode = MODE_BEGIN;
//            break;
//        }
//        break;
    }
  }
}
