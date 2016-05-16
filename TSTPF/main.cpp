
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
using namespace TST;
using namespace TSTPF;
using namespace PF;

#define MODE_TST_TRAIN 1
#define MODE_TST_PROPOSAL 2
#define MODE_TSTPF_TEST 3

#define EXPORT_BASE "/Users/beixinzhu/Desktop/movie_result/horses01_6/frame_"
#define EXPORT_EXTN ".png"

bool export_ = false;
//char* vid_file = "/Users/beixinzhu/Documents/UCLA/visionlab/testmovie/bag_ac_speed.mp4";
char* vid_file = "/Users/beixinzhu/Documents/dataset/movie_1.avi";
int get_regions( IplImage*);
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
    IplImage *frame;
    CvCapture* video;
    video = cvCaptureFromFile( vid_file );
    if( ! video ){
        printf("couldn't open video file %s", vid_file);
        return 1;
    }

    mode = MODE_TST_TRAIN;
    int TST_train_frame = 0;
    int TST_proposal_frame = 0;
    int TSTPF_test_frame = 0;
    int PFtimetoinit = false;
    int numframes = 0;

    while( frame = cvQueryFrame( video ) ){
        IplImage* framegrey = cvCreateImage(cvGetSize(frame), 8, 1);
        cvCvtColor(frame, framegrey, CV_BGR2GRAY);
        IplImage* hsv_frame;
        hsv_frame = bgr2hsv(frame);

        if (numframes == 0){
            num_objects = get_regions(frame);
        }
          printf("mode = %d\n",mode);

        switch (mode){
            case MODE_TST_TRAIN:
                numframes ++;TST_train_frame ++;
                printf("MODE_TRAIN, TST_train_frame / numframes = %d / %d\n",TST_train_frame,numframes);
//                TST_prep(frame,framegrey);
                // get started
                if (numframes == 1){
                    initialize_Detector(frame,framegrey);
                    TST_prep(frame,framegrey);
                    TST_RESET(frame,framegrey);
                }else
                    TST_prep(frame, framegrey);

                TST_TRAIN(frame, framegrey);

                // train PF
                if (TST_train_frame == 1)
                    PF_train(frame,hsv_frame);

                //check MAX_TST_TRAIN_FRAMES
                if (TST_train_frame >= MAX_TST_TRAIN_FRAMES)
                    mode =  MODE_TST_PROPOSAL;

                break;
            case MODE_TST_PROPOSAL:
                numframes ++;TST_proposal_frame ++;
                printf("MODE_TRAIN, TST_proposal_frame / numframes = %d / %d\n",TST_proposal_frame,numframes);
                TST_prep(frame, framegrey);
                TST_TEST(frame, framegrey);

                //check if can start track
                if (estm->istrack){
                    PF_init(frame,hsv_frame);
                    PF_test(frame, hsv_frame, framegrey);
                    mode = MODE_TSTPF_TEST;
                    break;
                }

                if (TST_proposal_frame > MAX_TST_PROPOSAL_FRAMES){
                    printf("Cannot generate enough proposals!\n");
                    return 0;
                }
                break;
            case MODE_TSTPF_TEST:
                numframes ++;TSTPF_test_frame ++;
                printf("MODE_TRAIN, TSTPF_test_frame / numframes = %d / %d\n",TSTPF_test_frame,numframes);
                TST_prep(frame, framegrey);
                TST_TEST(frame, framegrey);
                num_match = 1;

                PF_test(frame, hsv_frame, framegrey);
                break;
        }

        cvNamedWindow( "Video", 1 );
        cvShowImage( "Video", frame );

        if (export_)
            export_frame(frame,numframes);

        if(cvWaitKey( 5 ) == 27)
            break;
    //    cvDestroyAllWindows();
    //    cvReleaseCapture( &video );

  }
  return 0;
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


int get_regions( IplImage* frame )
{
    char* win_name = "First frame";
    params p;
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

    x1 = MIN( p.loc1[i].x, p.loc2[i].x );
    x2 = MAX( p.loc1[i].x, p.loc2[i].x );
    y1 = MIN( p.loc1[i].y, p.loc2[i].y );
    y2 = MAX( p.loc1[i].y, p.loc2[i].y );

    w = x2 - x1 + 1;
    h = y2 - y1 + 1;

    /* ensure odd width and height */
    w = ( w % 2 )? w : w + 1;
    h = ( h % 2 )? h : h + 1;

    regions = (CvRect*) malloc(1 * sizeof(CvRect));

    regions[0] = cvRect(x1, y1, w, h);

    return p.n;
}
//
//int get_regions( IplImage* frame, CvRect** regions )
//{
//    char* win_name = "First frame";
//    params p;
//    CvRect* r;
//    int i, x1, y1, x2, y2, w, h;
//    
//    /* use mouse callback to allow user to define object regions */
//    p.win_name = win_name;
//    p.orig_img = cvCloneImage( frame );
//    p.cur_img = NULL;
//    p.n = 0;
//    cvNamedWindow( win_name, 1 );
//    cvShowImage( win_name, frame );
//    cvSetMouseCallback( win_name, &mouse, &p );
//    cvWaitKey( 0 );
//    cvDestroyWindow( win_name );
//    cvReleaseImage( &(p.orig_img) );
//    if( p.cur_img )
//        cvReleaseImage( &(p.cur_img) );
//    
//    /* extract regions defined by user; store as an array of rectangles */
//    if( p.n == 0 )
//    {
//        *regions = NULL;
//        return 0;
//    }
//    r = (CvRect* )malloc( p.n * sizeof( CvRect ) );
//    for( i = 0; i < p.n; i++ )
//    {
//        x1 = MIN( p.loc1[i].x, p.loc2[i].x );
//        x2 = MAX( p.loc1[i].x, p.loc2[i].x );
//        y1 = MIN( p.loc1[i].y, p.loc2[i].y );
//        y2 = MAX( p.loc1[i].y, p.loc2[i].y );
//        w = x2 - x1;
//        h = y2 - y1;
//        
//        /* ensure odd width and height */
//        w = ( w % 2 )? w : w+1;
//        h = ( h % 2 )? h : h+1;
//        r[i] = cvRect( x1, y1, w, h );
//    }
//    *regions = r;
//    return p.n;
//}



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
