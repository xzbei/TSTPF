// File: globaldata.h
#ifndef globaldata_hpp
#define globaldata_hpp

//#include "defs.h"
//#include "particles.h"
//#include "observation.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "opencv/cv.h"
#include "TST.hpp"
#include "TrackMTD.hpp"
#include "observation.h"
#include "particles.h"

//using namespace GlobalConst;

namespace GlobalConst
{
    extern const int PARTICLES;
    extern const int MAX_OBJECTS;
    extern const float U0;
    extern const float U1;

    extern const int MODE_RESET;
    extern const int MODE_BEGIN;
    extern const int MODE_TRAIN;
    extern const int MODE_TEST;
    extern const float SCALE;
    extern const float THRES;
    extern const float PI;
    extern const int BTD_NUM_ROWS;
    extern const int BTD_NUM_COLS ;
    extern const int BTD_NUM_HISTOGRAMS ;
    extern const int BTD_NUM_BINS;
    extern const int BTD_SAMPLE_STEP;

    extern const int MTD_TRAIN_SCALES;
    extern const int MTD_NUM_SCALE;
    extern const int MTD_NUM_ANCHORS;
    extern const float MTD_SCORE_THRESHOLD;
    extern const bool MTD_PRUNE_TRACKING;
}

namespace GlobalVar
{
    extern bool mouseclick;
    extern int region[4];
    extern int num_particles;
    extern int show_all;
    extern IplImage* frame;
    // extern IplImage* framegrey;
    extern histogram** ref_histos;
    extern particle* particles;
    extern CvRect* regions;
//    CvRect* regions;
    extern int numframes;
    extern int num_objects;
    extern int nLevels;
    extern int max_num_feat;
    extern double threshold1;
    extern int mode;
    extern int width; //frame width
    extern int height; //frame height

    extern IplImage ** ppPyramid_curr;
    extern IplImage ** ppPyramid_prev;
    extern CvPoint2D32f * pFeat_curr;
    extern CvPoint2D32f * pFeat_prev;
    extern int num_feat;
    extern CvPoint * pFeat;
    extern int * pFeatScale;
    extern char * pFeatStatus;
    extern int * pow2;
    extern TrackMTD * trackMTD;
    extern int max_num_match;
    extern int * track_x;
    extern int * track_y;
    extern int * track_l;
    extern int * track_s;
    extern int * track_r;
    extern float * track_score;
    extern int num_match;
    extern int x1;
    extern int y1;
    extern float width_half;
    extern float height_half;
    extern bool TSTfirst_test;
    extern int visualize_num_intervals;
    extern int visualize_intervals2;
}

#endif /*globaldata_hpp*/
