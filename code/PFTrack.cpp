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
#include "unistd.h"
#include "opencv/cv.h"
#include <android/log.h>
#include "globaldata.hpp"

using namespace std;
using namespace cv;
using namespace GlobalConst;
using namespace GlobalVar;

float s;
int i, j, k, w, h, x, y,x2,y2,num,xcenter1,ycenter1,x0,y0,ww,hh;
int c = 0;
IplImage* hsv_frame;
CvScalar color;
double threshold1;
particle* new_particles;

void PF_RESET(){
  histogram** histos = malloc( num_objects * sizeof( histogram* ) );
  particles = malloc( num_particles * sizeof( particle ) );
  regions = malloc( num_objects * sizeof( CvRect ) );
}

particle* PF_train(IplImage* frame, Mat& mGr){
  hsv_frame = bgr2hsv(frame);
  CvRect* r;
  r = (CvRect*)malloc( 1 * sizeof( CvRect ) );

  ww = region[2] - region[0] + 1;
  hh = region[3] - region[1] + 1;
  ww = ( ww % 2 )? ww : ww+1;
  hh = ( hh % 2 )? hh : hh+1;
  r[0] = cvRect( region[0], region[1], ww, hh );
  regions = r;
  ref_histos = compute_ref_histos( hsv_frame, regions, num_objects );
  particles = init_distribution( regions, ref_histos, num_objects, num_particles , frame->width, frame->height, U0);

  num = calculate_alive(particles,num_particles);
  if( show_all )
      for( j = num_particles - 1; j > 0; j-- )
      {
          color = CV_RGB(255,255,0);
          x0 = round( particles[j].x - 0.5 * particles[j].s * particles[j].width );
          y0 = round( particles[j].y - 0.5 * particles[j].s * particles[j].height );
          x2 = x0 + round( particles[j].s * particles[j].width );
          y2 = y0 + round( particles[j].s * particles[j].height );
          xcenter1 = (x2 - x0)/2 -1;
          ycenter1 = (y2 - y0)/2 -1;
          rectangle( mGr, Point( (x0+xcenter1)*SCALE, (y0+ycenter1)*SCALE), Point( (x2-xcenter1)*SCALE, (y2-ycenter1)*SCALE ), color, 3, 8, 0 );
      }
      return particles;
}

void PF_test(IplImage* frame, Mat& mGr){
  hsv_frame = bgr2hsv(frame);
  for( j = 0; j < num_particles; j++ )
  {
      particles[j] = transition( particles[j], frame->width, frame->height, U0,U1,regions, ref_histos);
      s = particles[j].s;
      if (particles[j].alive == 1)
          particles[j].w = likelihood( hsv_frame, cvRound(particles[j].y),
                                       cvRound( particles[j].x ),
                                       cvRound( particles[j].width * s ),
                                       cvRound( particles[j].height * s ),
                                       particles[j].histo );
  }

  qsort( particles, num_particles, sizeof( particle ), &particle_cmp2 );

  normalize_weights( particles, num_particles );
  num = calculate_alive(particles,num_particles);
  new_particles = resample3(particles, num ,num_particles );

  particles = new_particles;

  qsort( particles, num_particles, sizeof( particle ), &particle_cmp );

  if( show_all )
      for( j = 0; j < num_particles; j++ )
      {
          if (particles[j].alive == 1)
          {
              color = CV_RGB(255, 255, 0);
              x0 = round(particles[j].x - 0.5 * particles[j].s * particles[j].width);
              y0 = round(particles[j].y - 0.5 * particles[j].s * particles[j].height);
              x2 = x0 + round(particles[j].s * particles[j].width);
              y2 = y0 + round(particles[j].s * particles[j].height);
              xcenter1 = (x2 - x0) / 2 - 1;
              ycenter1 = (y2 - y0) / 2 - 1;
              rectangle(mGr, Point((x0 + xcenter1)*SCALE, (y0 + ycenter1)*SCALE),
                        Point((x2 - xcenter1)*SCALE, (y2 - ycenter1)*SCALE), color, 3, 8, 0);
          }
      }

  particle center_particle = Meanshift_cluster(particles,num,80,frame->width, frame->height);
  x0 = round( center_particle.x - 0.5 * center_particle.s * center_particle.width );
  y0 = round( center_particle.y - 0.5 * center_particle.s * center_particle.height );
  x2 = x0 + round( center_particle.s * center_particle.width );
  y2 = y0 + round( center_particle.s * center_particle.height );
  x0 = MAX(0.0,x0);
  y0 = MAX(0.0,y0);
  x2 = MIN(frame->width,x2);
  y2 = MIN(frame->height,y2);

  double score1 = likelihood(hsv_frame,cvRound(center_particle.y),cvRound(center_particle.x),cvRound(center_particle.width*center_particle.s),
                             cvRound(center_particle.height*center_particle.s),center_particle.histo);
  __android_log_print(ANDROID_LOG_VERBOSE, "center_score","score1  = %f",score1);

  if (numframes == 2){
      threshold1 = score1*0.2;
      rectangle( mGr, Point( x0*SCALE, y0*SCALE), Point( x2*SCALE, y2*SCALE ), Scalar(255,0,0,255), 3, 8, 0 );
  }else {
      if (score1 >= threshold1)
          rectangle(mGr, Point(x0 * SCALE, y0 * SCALE), Point(x2 * SCALE, y2 * SCALE),
                    Scalar(255, 0, 0, 255), 3, 8, 0);
      else
          rectangle(mGr, Point(x0 * SCALE, y0 * SCALE), Point(x2 * SCALE, y2 * SCALE),
                    Scalar(255, 255, 0, 255), 3, 8, 0);
  }
}
