
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

#define EXPORT_BASE "/Users/beixinzhu/Desktop/movie_result/horses01_6/frame_"
#define EXPORT_EXTN ".png"

bool export_ = false;
CvScalar color;
//char* vid_file = "/Users/beixinzhu/Documents/UCLA/visionlab/testmovie/bag_ac_speed.mp4";
char* vid_file = "/Users/beixinzhu/Documents/dataset/movie_1.avi";
void on_mouse(int event, int x, int y, int flags, void* param );
int get_regions( IplImage*, CvRect** );
void mouse( int, int, int, int, void* );
int export_frame( IplImage* frame, int i );

typedef struct params {
    CvPoint loc1[1];
    CvPoint loc2[1];
    IplImage* objects[1];
    char* win_name;
    IplImage* orig_img;
    IplImage* cur_img;
    int n;
} params;

int main( int argc, char** argv )
{
  CvCapture* video;
  video = cvCaptureFromFile( vid_file );
    if( ! video ){
        printf("couldn't open video file %s", vid_file);
        return 1;
    }

  mode = MODE_TRAIN;

  while( frame = cvQueryFrame( video ) ){
    IplImage* framegrey = cvCreateImage(cvGetSize(frame), 8, 1);
    cvCvtColor(frame, framegrey, CV_BGR2GRAY);
    width = frame->width;
    height = frame->height;
    IplImage* hsv_frame;
    hsv_frame = bgr2hsv(frame);

    if (numframes == -1){
        num_objects = get_regions(frame,&regions);
    }
      printf("mode = %d\n",mode);

    switch (mode){
        case MODE_TRAIN:
            if (numframes == -1){
                initializeTracker(frame,framegrey);
                numframes = 0;
                num_feat = 0;
                TST_prep(frame,framegrey);
                TST_RESET(frame,framegrey);
                generating_proposal = false;
            }
//            TST_prep(frame,framegrey);
            TST_BEGIN(frame,framegrey);
            numframes ++;
            printf("MODE_TRAIN, numframes = %d\n",numframes);
            TST_train_frame ++;
            printf("TST_train_frame = %d  ",TST_train_frame);
            printf("TST_test_frame = %d\n",TST_test_frame);
            TST_prep(frame, framegrey);
            TST_TRAIN(frame, framegrey);
            if (TST_train_frame == 1)
                PF_train(frame,hsv_frame);
            if (TST_train_frame > 2){
                mode =  MODE_TEST;
                generating_proposal = true;
            }
            //            cvWaitKey(0);
            break;
        case MODE_TEST:
            printf("MODE_TEST, numframes = %d\n",numframes);
//            cvWaitKey(0);
            numframes ++;
            TST_prep(frame, framegrey);
            TST_TEST(frame, framegrey);
            printf("TST_train_frame = %d  ",TST_train_frame);
            printf("TST_test_frame = %d\n",TST_test_frame);
            if (generating_proposal){
                PFtimetoinit = Estimate_confidence();
            }
            num_match = 1;

            if (PFtimetotest){
              PF_test(frame, hsv_frame, framegrey);
//                cvWaitKey(0);
            }

            if (PFtimetoinit){
                PF_init(frame,hsv_frame);
                cvShowImage( "Video", frame );
//                cvWaitKey(0);
                PF_test(frame, hsv_frame, framegrey);
                cvShowImage( "Video", frame );
//                cvWaitKey(0);
                PFtimetoinit = false;
                PFtimetotest = true;
                generating_proposal = false;
             }
            mode =  MODE_TEST;
            break;
    }

    cvNamedWindow( "Video", 1 );

    cvShowImage( "Video", frame );

      if (export_)
          export_frame(frame,numframes);
//    cvSetMouseCallback("Video", &on_mouse,0);
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


int export_frame( IplImage* frame, int i )
{
  char name[ strlen(EXPORT_BASE) + strlen(EXPORT_EXTN) + 4 ];
  char num[5];

  snprintf( num, 5, "%04d", i );
  strcpy( name, EXPORT_BASE );
  strcat( name, num );
  strcat( name, EXPORT_EXTN );
  return cvSaveImage( name, frame ,0);
}


int get_regions( IplImage* frame, CvRect** regions )
{
    char* win_name = "First frame";
    params p;
    CvRect* r;
    int i, x1, y1, x2, y2, w, h;

    /* use mouse callback to allow user to define object regions */
    p.win_name = win_name;
    p.orig_img = cvCloneImage( frame );
    p.cur_img = NULL;
    p.n = 0;
    cvNamedWindow( win_name, 1 );
    cvShowImage( win_name, frame );
    cvSetMouseCallback( win_name, &mouse, &p );
    cvWaitKey( 0 );
    cvDestroyWindow( win_name );
    cvReleaseImage( &(p.orig_img) );
    if( p.cur_img )
        cvReleaseImage( &(p.cur_img) );

    /* extract regions defined by user; store as an array of rectangles */
    if( p.n == 0 )
    {
        *regions = NULL;
        return 0;
    }
    r = (CvRect* )malloc( p.n * sizeof( CvRect ) );
    for( i = 0; i < p.n; i++ )
    {
        x1 = MIN( p.loc1[i].x, p.loc2[i].x );
        x2 = MAX( p.loc1[i].x, p.loc2[i].x );
        y1 = MIN( p.loc1[i].y, p.loc2[i].y );
        y2 = MAX( p.loc1[i].y, p.loc2[i].y );
        w = x2 - x1;
        h = y2 - y1;

        /* ensure odd width and height */
        w = ( w % 2 )? w : w+1;
        h = ( h % 2 )? h : h+1;
        r[i] = cvRect( x1, y1, w, h );
    }
    *regions = r;
    return p.n;
}



/*
 Mouse callback function that allows user to specify the initial object
 regions.  Parameters are as specified in OpenCV documentation.
 */
void mouse( int event, int x, int y, int flags, void* param )
{
    params* p = (params*)param;
    CvPoint* loc;
    int n;
    IplImage* tmp;
    static int pressed = FALSE;

    /* on left button press, remember first corner of rectangle around object */
    if( event == CV_EVENT_LBUTTONDOWN )
    {
        n = p->n;
        if( n == MAX_OBJECTS )
            return;
        loc = p->loc1;
        loc[n].x = x;
        loc[n].y = y;
        pressed = TRUE;
    }

    /* on left button up, finalize the rectangle and draw it in black */
    else if( event == CV_EVENT_LBUTTONUP )
    {
        n = p->n;
        if( n == MAX_OBJECTS )
            return;
        loc = p->loc2;
        loc[n].x = x;
        loc[n].y = y;
        cvReleaseImage( &(p->cur_img) );
        p->cur_img = NULL;
        cvRectangle( p->orig_img, p->loc1[n], loc[n], CV_RGB(0,0,0), 1, 8, 0 );
        cvShowImage( p->win_name, p->orig_img );
        pressed = FALSE;
        p->n++;
    }

    /* on mouse move with left button down, draw rectangle as defined in white */
    else if( event == CV_EVENT_MOUSEMOVE  &&  flags & CV_EVENT_FLAG_LBUTTON )
    {
        n = p->n;
        if( n == MAX_OBJECTS )
            return;
        tmp = cvCloneImage( p->orig_img );
        loc = p->loc1;
        cvRectangle( tmp, loc[n], cvPoint(x, y), CV_RGB(255,255,255), 1, 8, 0 );
        cvShowImage( p->win_name, tmp );
        if( p->cur_img )
            cvReleaseImage( &(p->cur_img) );
        p->cur_img = tmp;
    }
}
