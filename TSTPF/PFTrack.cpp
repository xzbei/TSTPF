//
//  PFTrack.cpp
//  TSTPF
//
//  Created by BeiXinzhu on 4/7/16.
//  Copyright © 2016 BeiXinzhu. All rights reserved.
//

#include "PFTrack.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "defs.h"
#include "particles.h"
#include "observation.h"
#include "time.h"
//#include "unistd.h"
#include "opencv/cv.h"
#include "globaldata.hpp"

using namespace std;
using namespace cv;
using namespace TST;
using namespace TSTPF;
using namespace PF;


int i, j, k, w, h, x, y,num,ww,hh;
double score1;

void PF_train(IplImage* frame, IplImage* hsv_frame){
    cvRectangle(frame, cvPoint(regions[0].x, regions[0].y), cvPoint(regions[0].x + regions[0].width, regions[0].y + regions[0].height), CV_RGB(0, 0, 255));
    ref_histos = compute_ref_histos( hsv_frame, regions, num_objects );
}

particle* PF_init(IplImage* pImageFrame, IplImage* hsv_frame){
    particles = init_distribution( ref_histos, num_particles, pImageFrame->width, pImageFrame->height, U0, *estm);
    int i;
    for (i = 0;i<num_particles; i++){
        if (particles[i].alive == 1){
            cvCircle(pImageFrame, cvPoint(particles[i].x, particles[i].y), 2, CV_RGB(0, 255, 255),-1);
        }
    }
    num = calculate_alive(particles,num_particles);
    printf("DONE Initialization...Initialized %d particles..\n",num);

    score1 = likelihood1(hsv_frame,cvRound(estm->pos_y[estm->pointer]),cvRound(estm->pos_x[estm->pointer]),cvRound(estm->width_record[estm->pointer]),
                               cvRound(estm->width_record[estm->pointer]),ref_histos[0],false);
    estm_PF = new estimate();
    estm_PF = init_Estimate(estm_PF, MAX_PF_RECORD_FRAMES, estm->pos_x[estm->pointer], estm->pos_y[estm->pointer], estm->width_record[estm->pointer],
        estm->width_record[estm->pointer], score1);
    return particles;
}

void PF_test(IplImage* frame, IplImage* hsv_frame, IplImage* framegrey){
    float sum_score;
    int x[num_match], y[num_match];
    float np[num_match];
    int width[num_match];
    int height[num_match];
    for (i = 0; i < num_match; i++){
      sum_score += track_score[i];
    }
    np[0] = track_score[0] / sum_score;
    for (i = 1; i < num_match; i++){
      np[i] = np[i-1] + track_score[i] / sum_score;
    }

    for (i = 0; i < num_match; i++){
      width[i] = regions[i].width;
      height[i] = regions[i].height;
      x[i] = regions[i].x + width[i] / 2;
      y[i] = regions[i].y + height[i] / 2;
    }

    for( j = 0; j < num_particles; j++ )
    {
        particles[j] = transition( particles[j], frame->width, frame->height, U0, U1, ref_histos, *estm, *estm_PF);

        float sw,sh;
        sw = particles[j].sw;
        sh = particles[j].sh;
        if (particles[j].alive == 1){
            particles[j].w = likelihood1( hsv_frame, cvRound(particles[j].y), cvRound( particles[j].x ), cvRound( particles[j].width * sw ), cvRound( particles[j].height * sh ), particles[j].histo ,true);
//             particles[j].w =likelihood2(framegrey, cvRound(particles[j].y), cvRound( particles[j].x ), cvRound( particles[j].width * s ), cvRound( particles[j].height * s ), ppPyramid_curr, trackMTD->mtd, 100000);
            if (particles[j].w >= 100000){
                particles[j].alive = 0;
            }
        }
    }

    qsort( particles, num_particles, sizeof( particle ), &particle_cmp2 );// put alive particles to the front

    num = normalize_weights( particles, num_particles );
    printf("alive particles %d\n",num);
    particle* new_particles;
    new_particles = resample3(particles, num ,num_particles );
    particles = new_particles;
//    free(particles);

    qsort( particles, num_particles, sizeof( particle ), &particle_cmp );

    if( show_all ){
      for( j = num_particles - 1; j > 0; j-- )
      {
        if (particles[j].alive == 1){
          CvScalar color;
          color = CV_RGB(255,255,0);
          cvCircle( frame,cvPoint(particles[j].x,particles[j].y),2,color,-1,1,0);
        }
      }
    }

    particle center_particle = Meanshift_cluster(particles,num,frame->width / 2,frame->width, frame->height);
    display_boundingbox(frame,center_particle, CV_RGB(255, 0, 0), 2,8,0);

    // is_track parameter need to corporate with PN classifier
    score1 = likelihood1(hsv_frame,cvRound(estm->pos_y[estm->pointer]),cvRound(estm->pos_x[estm->pointer]),cvRound(estm->width_record[estm->pointer]),
                               cvRound(estm->width_record[estm->pointer]),ref_histos[0],true);
    set_Estimate(estm_PF,true, estm_PF->pointer + 1, center_particle.x, center_particle.y, center_particle.sw * center_particle.width,
        center_particle.sh * center_particle.height, score1,MAX_PF_RECORD_FRAMES,MIN_PF_RECORD_FRAMES);

//    visualize_particle_heatmap(frame, particles, num_particles, visualize_num_intervals, num);
//    visualize_particle_heatmap2(frame, particles, num_particles, visualize_intervals2, num);
}

float likelihood2( IplImage* img, int r, int c,
                  int w, int h, IplImage ** pyramid, MTD ** mtd ,float threshold)
{
    float match_score1 = MTD_test( mtd[0], (unsigned char *)(pyramid[0]->imageData), pyramid[0]->widthStep, w, h, c, r, 0, threshold );
    float match_score2 = motion_likelihood(c,r,w,h);
    return exp(- LAMBDA1 * LAMBDA2 * match_score1 - LAMBDA1 * (1 - LAMBDA2) * match_score2);
}

float likelihood1( IplImage* img, int r, int c, int w, int h, histogram* ref_histo, bool motion_like)
{
    float match_score1 = likelihood(img,r,c,w,h,ref_histo);
    if (motion_like){
        float match_score2 = motion_likelihood(c,r,w,h);
        return exp(- LAMBDA1 * LAMBDA2 * match_score1 - LAMBDA1 * (1 - LAMBDA2) * match_score2);
    }else{
        return exp(- LAMBDA1 * match_score1);
    }
    
}

float motion_likelihood(int x, int y, int w,int h){
    float match_score = 0;
    match_score += 1.3 * abs(w - estm_PF->width);
    match_score += 1.3 * abs(h - estm_PF->height);
//    match_score += 0.3 * abs((x - estm_PF->pos_x[0]) - estm_PF->velocity_x);
//    match_score += 0.3 * abs((y - estm_PF->pos_y[0]) - estm_PF->velocity_y);
    match_score += 1.2 * abs(x - estm_PF->pos_x[estm_PF->pointer] - estm_PF->velocity_x);
    match_score += 1.2 * abs(y - estm_PF->pos_y[estm_PF->pointer] - estm_PF->velocity_y);
    match_score += 1.5 * abs(x - estm_PF->pos_x[estm_PF->pointer]);
    match_score += 1.5 * abs(y - estm_PF->pos_y[estm_PF->pointer]);
    match_score += 2 * fabsf((float)w / (float)h - (float)BTD_NUM_BINS / (float)BTD_NUM_COLS);

    return match_score/1000;
}

void display_boundingbox(IplImage* img,particle p, CvScalar color, int a, int b ,int c){
    cvCircle(img, CvPoint(p.x,p.y), 4, color);
    float width_half = 0.5 * p.sw * p.width;
    float height_half = 0.5 * p.sh * p.height;
    int x1 = round( p.x - width_half );
    int y1 = round( p.y - height_half );
    int x2 = round( p.x + width_half );
    int y2 = round( p.y + height_half );

    x1 = MAX(0.0,x1);
    y1 = MAX(0.0,y1);
    x2 = MIN(img->width,x2);
    y2 = MIN(img->height,y2);
    cvRectangle(img, Point( x1,y1), Point( x2,y2), color, a, b, c);
}
