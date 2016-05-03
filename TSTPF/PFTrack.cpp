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
using namespace GlobalConst;
using namespace GlobalVar;


int i, j, k, w, h, x, y,x2,y2,num,xcenter1,ycenter1,x3,y3,ww,hh;


//float threshold1;

//void PF_RESET(){
//    histogram** histos = malloc( num_objects * sizeof( histogram* ) );
//    particles = malloc( num_particles * sizeof( particle ) );
//    regions = malloc( num_objects * sizeof( CvRect ) );
//}

void PF_train(IplImage* frame, IplImage* hsv_frame){

  CvRect* r;
  r = (CvRect*)malloc( 1 * sizeof( CvRect ) );

  ww = region[2] - region[0] + 1;
  hh = region[3] - region[1] + 1;
  ww = ( ww % 2 )? ww : ww+1;
  hh = ( hh % 2 )? hh : hh+1;
  r[0] = cvRect( region[0], region[1], ww, hh );
    cvRectangle(frame, cvPoint(region[0], region[1]), cvPoint(region[2], region[3]), CV_RGB(0, 0, 255));
  regions = r;
  ref_histos = compute_ref_histos( hsv_frame, regions, num_objects );
}

particle* PF_init(IplImage* pImageFrame, IplImage* hsv_frame){
    particles = init_distribution( regions, ref_histos, num_match, num_particles , frame->width, frame->height, U0, track_score, num_match, *estm);

    int i;
    for (i = 0;i<num_particles; i++){
        if (particles[i].alive == 1){
            cvCircle(frame, cvPoint(particles[i].x, particles[i].y), 5, CV_RGB(0, 255, 255),-1);
        }
    }
    num = calculate_alive(particles,num_particles);
    // printf("alive particles %d\n",num);

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
        int aa = particles[j].alive;
        particles[j] = transition( particles[j], frame->width, frame->height, U0,U1,regions, ref_histos, np, x, y, width, height, num_match,*estm);
//        if (aa == 0 && particles[j].alive == 1){
//            cvCircle(frame, cvPoint(particles[j].x, particles[j].y), 2, CV_RGB(0, 255, 255),-1);
//        }//born
//
//        if (aa == 1 && particles[j].alive == 1){
//            cvCircle(frame, cvPoint(particles[j].x, particles[j].y), 2, CV_RGB(255, 100, 255),-1);
//        }//maintain

        float s;
        s = particles[j].s;
        if (particles[j].alive == 1){
            particles[j].w = likelihood1( hsv_frame, cvRound(particles[j].y), cvRound( particles[j].x ), cvRound( particles[j].width * s ), cvRound( particles[j].height * s ), particles[j].histo );
//             particles[j].w =likelihood2(framegrey, cvRound(particles[j].y), cvRound( particles[j].x ), cvRound( particles[j].width * s ), cvRound( particles[j].height * s ), ppPyramid_curr, trackMTD->mtd, 100000);
            if (particles[j].w == -1){
                particles[j].alive = 0;
            }
        }
    }

    qsort( particles, num_particles, sizeof( particle ), &particle_cmp2 );

    normalize_weights( particles, num_particles );
    num = calculate_alive(particles,num_particles);
    printf("alive particles %d\n",num);
    particle* new_particles;
    new_particles = resample3(particles, num ,num_particles );

    particles = new_particles;

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
    cvCircle(frame, CvPoint(center_particle.x,center_particle.y), 4, CV_RGB(255, 0, 0));
    x3 = round( center_particle.x - 0.5 * center_particle.s * center_particle.width );
    y3 = round( center_particle.y - 0.5 * center_particle.s * center_particle.height );
    x2 = x3 + round( center_particle.s * center_particle.width );
    y2 = y3 + round( center_particle.s * center_particle.height );
    x3 = MAX(0.0,x3);
    y3 = MAX(0.0,y3);
    x2 = MIN(frame->width,x2);
    y2 = MIN(frame->height,y2);

    double score1 = likelihood1(hsv_frame,cvRound(center_particle.y),cvRound(center_particle.x),cvRound(center_particle.width*center_particle.s),
                               cvRound(center_particle.height*center_particle.s),center_particle.histo);

//    if (numframes == 2){

    if (PFtimetoinit){
        threshold1 = score1*0.05;
        cvRectangle( frame, Point( x3*SCALE, y3*SCALE), Point( x2*SCALE, y2*SCALE ), CV_RGB(255, 0, 0), 3, 8, 0);
        estm_PF->istrack = true;
        estm_PF->velocity_x = estm->velocity_x;
        estm_PF->velocity_y = estm->velocity_y;
    }else {
        estm_PF->velocity_x = center_particle.x - estm_PF->pos_x[0];
        estm_PF->velocity_y = center_particle.y - estm_PF->pos_y[0];
        if (score1 >= threshold1){
            estm_PF->istrack = true;
            cvRectangle(frame, Point(x3 * SCALE, y3 * SCALE), Point(x2 * SCALE, y2 * SCALE),
                      CV_RGB(255, 0, 0) , 3, 8, 0);
                  }
        else{
            estm_PF->istrack = false;
            cvRectangle(frame, Point(x3 * SCALE, y3 * SCALE), Point(x2 * SCALE, y2 * SCALE),
                      CV_RGB(0, 255, 0), 3, 8, 0);
                  }
    }

    estm_PF->pos_x[0] = center_particle.x;
    estm_PF->pos_y[0] = center_particle.y;
    estm_PF->width = center_particle.s * center_particle.width;
    estm_PF->height = center_particle.s * center_particle.height;

//    visualize_particle_heatmap(frame, particles, num_particles, visualize_num_intervals, num);
//    visualize_particle_heatmap2(frame, particles, num_particles, visualize_intervals2, num);
}

float likelihood2( IplImage* img, int r, int c,
                  int w, int h, IplImage ** pyramid, MTD ** mtd ,float threshold)
{
    float match_score1 = MTD_test( mtd[0], (unsigned char *)(pyramid[0]->imageData), pyramid[0]->widthStep, w, h, c, r, 0, threshold );
    if (PFtimetoinit)
        return exp( -LAMBDA1 * match_score1 );
    else{
        float match_score2 = motion_likelihood(c,r,w,h);
        return exp(- LAMBDA1 * LAMBDA2 * match_score1 - LAMBDA1 * (1 - LAMBDA2) * match_score2);
    }
}

float likelihood1( IplImage* img, int r, int c, int w, int h, histogram* ref_histo)
{
    float match_score1 = likelihood(img,r,c,w,h,ref_histo);
    if (PFtimetoinit)
        return exp( -LAMBDA1 * match_score1 );
    else{
        float match_score2 = motion_likelihood(c,r,w,h);
        return exp(- LAMBDA1 * LAMBDA2 * match_score1 - LAMBDA1 * (1 - LAMBDA2) * match_score2);
    }
}

float motion_likelihood(int x, int y, int w,int h){
    float match_score = 0;
    match_score += 1 * abs(w - estm_PF->width);
    match_score += 1 * abs(h - estm_PF->height);
//    match_score += 0.3 * abs((x - estm_PF->pos_x[0]) - estm_PF->velocity_x);
//    match_score += 0.3 * abs((y - estm_PF->pos_y[0]) - estm_PF->velocity_y);
    match_score += 1.2 * abs(x - estm_PF->pos_x[0] - estm_PF->velocity_x);
    match_score += 1.2 * abs(y - estm_PF->pos_y[0] - estm_PF->velocity_y);
    match_score += 1.5 * abs(x - estm_PF->pos_x[0]);
    match_score += 2.5 * abs(y - estm_PF->pos_y[0]);

    return match_score/1000;
}
