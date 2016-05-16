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

namespace TSTPF{
    extern const float RESART_TRAIN_THRES ;
    extern const float START_TRACKING_THRES ;
    extern const float MAX_TST_PROPOSAL_FRAMES ;
    extern const float MIN_TST_PROPOSAL_FRAMES ;
    extern const int MAX_TST_TRAIN_FRAMES;
    extern const int MAX_PF_RECORD_FRAMES;
    extern const int MIN_PF_RECORD_FRAMES;

    extern int mode;
    extern CvRect* regions;
    extern estimate* estm;
    extern estimate* estm_PF;
    static estimate* init_Estimate(estimate* estm, int max_nframe, float init_x, float init_y, int init_width, int init_height, float init_score){
        //    estm = new estimate();
        estm->pos_x = (float *)malloc (sizeof(float) * max_nframe);
        estm->pos_y = (float *)malloc (sizeof(float) * max_nframe);
        estm->width_record = (int *)malloc (sizeof(int) * max_nframe);
        estm->height_record = (int *)malloc (sizeof(int) * max_nframe);
        estm->score = (float *)malloc( sizeof(float) * max_nframe );
        estm->valid = (int *)malloc (sizeof(int) * max_nframe);
        memset(estm->valid, 0, (size_t)max_nframe);
        estm->valid[0] = 1;
        estm->pos_x[0] = init_x;
        estm->pos_y[0] = init_y;
        estm->width_record[0] = init_width;
        estm->height_record[0] = init_height;
        estm->score[0] = init_score;
        estm->istrack = false;
        estm->pointer = 0;
        return estm;
    }
    
    static int set_Estimate(estimate* estm,bool is_track, int pointer = 0, float x = 0, float y = 0, int w = 0, int h = 0, float score = 0.0, int max_nframe = MAX_TST_PROPOSAL_FRAMES, int min_nframe = MIN_TST_PROPOSAL_FRAMES){
        int i;
        
        pointer = pointer % max_nframe;
        
        estm->pos_x[pointer] = x;
        estm->pos_y[pointer] = y;
        estm->width_record[pointer] = w;
        estm->height_record[pointer] = h;
        estm->score[pointer] = score;
        estm->pointer = pointer;
        
        if (!is_track){
            estm->valid[pointer] = -1;
        }else {
            estm->valid[pointer] = 1;
        }
        
        estm->istrack = true;
        
        for (i = pointer; i > pointer - min_nframe; --i )
            if (estm->valid[(i + max_nframe) % max_nframe] == -1){
                estm->istrack = false;
                return 0;
                break;
            }
        
        int counter1 = 0, counter2 = 0, counter3 = 0;
        int curr,prev,pprev;
        float width = 0,height = 0,velocity_x = 0,velocity_y = 0, accelerate_vx = 0, accelerate_vy = 0;
        for (i = pointer; i > pointer - min_nframe; --i){
            curr = (i + max_nframe) % max_nframe;
            prev = (i - 1 + max_nframe) % max_nframe;
            pprev = (i - 2 + max_nframe) % max_nframe;
            
            if (estm->valid[curr] == 1){
                counter1++;
                width += estm->width_record[curr];
                height += estm->height_record[curr];
                if (estm->valid[prev] == 1){
                    counter2++;
                    velocity_x += estm->pos_x[curr] - estm->pos_x[prev];
                    velocity_y += estm->pos_y[curr] - estm->pos_y[prev];
            
                    if (estm->valid[pprev] == 1){
                        counter3++;
                        accelerate_vx += estm->pos_x[curr] - 2 * estm->pos_x[prev] + estm->pos_x[pprev];
                        accelerate_vy += estm->pos_y[curr] - 2 * estm->pos_y[prev] + estm->pos_y[pprev];
                    }
                }
            }
        }
        
        estm->width = counter1 == 0? estm->width : width / (float)counter1;
        estm->height = counter1 == 0 ? estm->height : height / (float)counter1;
        estm->velocity_x = counter2 == 0 ? estm->velocity_x : velocity_x / (float)counter2;
        estm->velocity_y = counter2 == 0 ? estm->velocity_y : velocity_y / (float)counter2;
        estm->accelerate_vx = counter3 == 0 ? estm->accelerate_vx : accelerate_vx / (float)counter3;
        estm->accelerate_vy = counter3 == 0 ? estm->accelerate_vy : accelerate_vy / (float)counter3;
        
//        estm->velocity_x += ( min_nframe / 2) * estm->accelerate_vx;
//        estm->velocity_y += (min_nframe / 2) * estm->accelerate_vy;
        
        return 1;
    }
}

namespace TST{
    extern const float SCALE ;
    extern const float THRES ;
    extern const float PI ;
    extern const int BTD_NUM_ROWS;
    extern const int BTD_NUM_COLS;
    extern const int BTD_NUM_HISTOGRAMS;
    extern const int BTD_NUM_BINS;
    extern const int BTD_SAMPLE_STEP;

    extern const int MTD_TRAIN_SCALES;
    extern const int MTD_NUM_SCALE;
    extern const int MTD_NUM_ANCHORS;
    extern const float MTD_SCORE_THRESHOLD;
    extern const bool MTD_PRUNE_TRACKING;

    extern int nLevels;
    extern int max_num_feat;
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
}

namespace PF{
    extern const int PARTICLES;
    extern const int MAX_OBJECTS;
    extern const float U0;
    extern const float U1;

    extern int num_particles;
    extern int show_all;
    extern histogram** ref_histos;
    extern particle* particles;
    extern int num_objects;
    extern double threshold1;

    extern int visualize_num_intervals;
    extern int visualize_intervals2;

}



#endif /*globaldata_hpp*/
