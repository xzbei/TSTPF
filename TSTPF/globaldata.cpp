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
#include "globaldata.hpp"
#include "observation.h"
#include "particles.h"

using namespace TSTPF;

namespace TSTPF{
    const float RESART_TRAIN_THRES = 0.2;
    const float START_TRACKING_THRES = 0.5;
    const float MAX_TST_PROPOSAL_FRAMES = 20;
    const float MIN_TST_PROPOSAL_FRAMES = 3;
    const int MAX_TST_TRAIN_FRAMES = 3;
    const int MAX_PF_RECORD_FRAMES = 20;
    const int MIN_PF_RECORD_FRAMES = 4;

    int mode;
    CvRect* regions;
    estimate* estm;
    estimate* estm_PF;
}

namespace TST{
    const float SCALE = 1;
    const float THRES = 0.05;
    const float PI = 3.14159265;
    const int BTD_NUM_ROWS    = 12;
//    const int BTD_NUM_ROWS    = 30;
    const int BTD_NUM_COLS    = 12;
//    const int BTD_NUM_COLS    = 30;
    const int BTD_NUM_HISTOGRAMS  = BTD_NUM_ROWS * BTD_NUM_COLS;
    const int BTD_NUM_BINS    = 8;
    const int BTD_SAMPLE_STEP   = 1;
//    const int BTD_SAMPLE_STEP   = 3;

    const int MTD_TRAIN_SCALES  = 3;
    const int MTD_NUM_SCALE   = 3;
    const int MTD_NUM_ANCHORS   = 3;
//    const float MTD_SCORE_THRESHOLD = 0.2;
    const float MTD_SCORE_THRESHOLD = 0.3;
    const bool MTD_PRUNE_TRACKING = true;

    int nLevels = 5;
    int max_num_feat = 500;
    IplImage ** ppPyramid_curr;
    IplImage ** ppPyramid_prev;
    CvPoint2D32f * pFeat_curr;
    CvPoint2D32f * pFeat_prev;
    int num_feat;
    CvPoint * pFeat;
    int * pFeatScale;
    char * pFeatStatus;
    int * pow2;
    TrackMTD * trackMTD;
    int max_num_match  = 1000;
    int * track_x;
    int * track_y;
    int * track_l;
    int * track_s;
    int * track_r;
    float * track_score;
    int num_match;

}

namespace PF{
    const int PARTICLES = 5000;
    const int MAX_OBJECTS = 1;
    const float U0 = 0.2;
    const float U1 = 0.2;

    int num_particles = PARTICLES;
    int show_all = 1;
    histogram** ref_histos;
    particle* particles;
    int num_objects = 1;
    double threshold1;

    int visualize_num_intervals = 20;
    int visualize_intervals2 = 50;

}



