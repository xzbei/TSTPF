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
  regions = r;
  ref_histos = compute_ref_histos( hsv_frame, regions, num_objects );
}

particle* PF_init(IplImage* pImageFrame, IplImage* hsv_frame){
    particles = init_distribution( regions, ref_histos, num_match, num_particles , frame->width, frame->height, U0, track_score, num_match);
    num = calculate_alive(particles,num_particles);
    printf("alive particles %d\n",num);

    return particles;
}

void PF_test(IplImage* frame, IplImage* hsv_frame, bool & TSTfirst_test){

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
//      if (!TSTfirst_test){
        particles[j] = transition( particles[j], frame->width, frame->height, U0,U1,regions, ref_histos, np, x, y, width, height, num_match);
//      }
        float s;
        s = particles[j].s;
        if (particles[j].alive == 1){
            particles[j].w = likelihood( hsv_frame, cvRound(particles[j].y),
                                        cvRound( particles[j].x ),
                                        cvRound( particles[j].width * s ),
                                        cvRound( particles[j].height * s ),
                                        particles[j].histo );
            if (particles[j].w == -1){
                particles[j].alive = 0;
            }
        }
    }
//    if (!TSTfirst_test) TSTfirst_test = false;

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

    particle center_particle = Meanshift_cluster(particles,num,50,frame->width, frame->height);
//    center_particle.width = width[0];
//    center_particle.height = height[0];
    x3 = round( center_particle.x - 0.5 * center_particle.s * center_particle.width );
    y3 = round( center_particle.y - 0.5 * center_particle.s * center_particle.height );
//     x3 = round( center_particle.x);
//     y3 = round( center_particle.y);
    x2 = x3 + round( center_particle.s * center_particle.width );
    y2 = y3 + round( center_particle.s * center_particle.height );
    x3 = MAX(0.0,x3);
    y3 = MAX(0.0,y3);
    x2 = MIN(frame->width,x2);
    y2 = MIN(frame->height,y2);

    double score1 = likelihood(hsv_frame,cvRound(center_particle.y),cvRound(center_particle.x),cvRound(center_particle.width*center_particle.s),
                               cvRound(center_particle.height*center_particle.s),center_particle.histo);

//    if (numframes == 2){
    if (TSTfirst_test){
        threshold1 = score1*0.1;
        printf("threshold = %f\n",threshold1);
        cvRectangle( frame, Point( x3*SCALE, y3*SCALE), Point( x2*SCALE, y2*SCALE ), CV_RGB(255, 0, 0), 3, 8, 0 );
    }else {
        if (score1 >= threshold1)
            cvRectangle(frame, Point(x3 * SCALE, y3 * SCALE), Point(x2 * SCALE, y2 * SCALE),
                      CV_RGB(255, 0, 0) , 3, 8, 0);
        else
            cvRectangle(frame, Point(x3 * SCALE, y3 * SCALE), Point(x2 * SCALE, y2 * SCALE),
                      CV_RGB(0, 255, 0), 3, 8, 0);
    }
}
