#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <algorithm>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include <android/log.h>

#include "TST.h"
#include "TrackMTD.h"
#include "TSTRecognizer.hpp"
#include "globaldata.hpp"

using namespace cv;
using namespace std;
using namespace GlobalVar;
using namespace GlobalConst;

bool DescendingOrder(cv::KeyPoint pt1, cv::KeyPoint pt2) {
    return (pt1.response > pt2.response);
}

// ######################################################################
void initializeTracker(IplImage* pImageFrame, IplImage* pImageGray)
{
  pow2 = (int *)malloc( sizeof(int) * 10 );
  pow2[0] = 1;
  for ( int i = 1 ; i < 10 ; i ++ ) pow2[i] = 2 * pow2[i-1];

  ppPyramid_curr = (IplImage **)malloc( sizeof(IplImage *) * nLevels );
  ppPyramid_prev = (IplImage **)malloc( sizeof(IplImage *) * nLevels );
  pFeat_curr = (CvPoint2D32f *)malloc( sizeof(CvPoint2D32f) * max_num_feat );
  pFeat_prev = (CvPoint2D32f *)malloc( sizeof(CvPoint2D32f) * max_num_feat );

  // init MTD
  trackMTD = new TrackMTD();
  trackMTD->Initialize( BTD_NUM_ROWS, BTD_NUM_COLS, BTD_NUM_BINS, BTD_SAMPLE_STEP, MTD_NUM_SCALE, MTD_NUM_ANCHORS, width, height );
  track_x = (int *)malloc( sizeof(int) * max_num_match );
  track_y = (int *)malloc( sizeof(int) * max_num_match );
  track_l = (int *)malloc( sizeof(int) * max_num_match );
  track_s = (int *)malloc( sizeof(int) * max_num_match );
  track_r = (int *)malloc( sizeof(int) * max_num_match );
  track_score = (float *)malloc( sizeof(float) * max_num_match );
  num_match = 0;
  track_x[0] = width/2;
  track_y[0] = height/2;
  track_l[0] = MTD_TRAIN_SCALES;
  track_s[0] = 0;
  track_r[0] = 0;

  pFeat = (CvPoint *)malloc( sizeof(CvPoint) * max_num_feat );
  pFeatScale = (int *)malloc( sizeof(int) * max_num_feat );
  pFeatStatus = (char *)malloc( sizeof(char) * max_num_feat );

  x1 = width/2;
  y1 = height/2;
  region[0] = x1 - pow2[MTD_TRAIN_SCALES]*BTD_NUM_COLS*BTD_SAMPLE_STEP/2;
  region[1] = y1 - pow2[MTD_TRAIN_SCALES]*BTD_NUM_ROWS*BTD_SAMPLE_STEP/2;
  region[2] = x1 + pow2[MTD_TRAIN_SCALES]*BTD_NUM_COLS*BTD_SAMPLE_STEP/2;
  region[3] = y1 + pow2[MTD_TRAIN_SCALES]*BTD_NUM_ROWS*BTD_SAMPLE_STEP/2;

  init_image_pyramid( pImageGray, ppPyramid_curr, nLevels );
  init_image_pyramid( pImageGray, ppPyramid_prev, nLevels );

  mode = MODE_BEGIN;
  return;
}

void TST_prep(IplImage* pImageFrame, IplImage* pImageGray){
  if(ppPyramid_prev[0]->depth != 8)
  {
    cout<<"BAD: "<<ppPyramid_prev[0]->depth<<" "<<ppPyramid_prev[0]->nChannels<<endl;
    init_image_pyramid( pImageGray, ppPyramid_prev, nLevels );
  }

  IplImage ** pyr_temp = ppPyramid_curr;
  ppPyramid_curr = ppPyramid_prev;
  ppPyramid_prev = pyr_temp;
  CvPoint2D32f * feat_temp = pFeat_curr;
  pFeat_curr = pFeat_prev;
  pFeat_prev = feat_temp;

  build_image_pyramid( pImageGray, ppPyramid_curr, nLevels );
}

void TST_RESET(IplImage* pImageFrame, IplImage* pImageGray){
  MTD_free( &(trackMTD->mtd[0]) );
  trackMTD->mtd[0] = MTD_create( trackMTD->num_templates, trackMTD->num_anchors, trackMTD->mtd_filt );
  num_match = 0;
  track_x[0] = width/2;
  track_y[0] = height/2;
  track_l[0] = MTD_TRAIN_SCALES;
  track_s[0] = 0;
  track_r[0] = 0;
  x1 = width/2;
  y1 = height/2;
}

void TST_BEGIN(IplImage* pImageFrame, IplImage* pImageGray){
  width_half  = trackMTD->get_mtd_width ( MTD_TRAIN_SCALES, 0 ) / 2;
  height_half = trackMTD->get_mtd_height( MTD_TRAIN_SCALES, 0 ) / 2;
  region[0] = track_x[0]-width_half;
  region[1] = track_y[0]-height_half;
  region[2] = track_x[0]+width_half;
  region[3] = track_y[0]+height_half;
  cvRectangle(pImageFrame, cvPoint(x1-width_half,y1-height_half), cvPoint(x1+width_half,y1+height_half), CV_RGB(255,255,255), 3, 1, 0);
}

void TST_TRAIN(IplImage* pImageFrame, IplImage* pImageGray){
  num_feat = 0;
  if ( num_feat == 0 )
  {
    for ( int i = nLevels - 1 ; i >= 0 ; i -- )
    {
      if ( num_feat >= max_num_feat ) break;

              vector<KeyPoint> keypoints;
              Mat * temp = new Mat(ppPyramid_prev[i]);
              FAST( *temp, keypoints, 10 );
              int num_new_corners = keypoints.size();
              if (num_new_corners == 0) continue;
              sort( keypoints.begin(), keypoints.end(), DescendingOrder);
      for ( int j = 0 ; j < num_new_corners && num_feat < max_num_feat ; j ++ )
      {
                  float x = keypoints[j].pt.x;
                  float y = keypoints[j].pt.y;

        if ( x < 10 || x >= ppPyramid_curr[i]->width-10 ||
           y < 10 || y >= ppPyramid_curr[i]->height-10 )
          continue;

        // add features
        pFeat[num_feat] = cvPoint(x, y);
        pFeat_prev[num_feat].x = x;
        pFeat_prev[num_feat].y = y;
        pFeatScale[num_feat] = i;
        pFeatStatus[num_feat] = 1;
        num_feat ++;
      }
    }
  }
  // track features and train BTD
  if ( num_feat > 0 )
  {
    trackTST( ppPyramid_prev, ppPyramid_curr, pFeat_prev, pFeat_curr, pFeatScale, num_feat, pFeatStatus );

    // get the center x and y
    float good_count = 0;
    float dx = 0;
    float dy = 0;
    float width_half  = trackMTD->get_mtd_width ( MTD_TRAIN_SCALES, 0 ) / 2;
    float height_half = trackMTD->get_mtd_height( MTD_TRAIN_SCALES, 0 ) / 2;
    for ( int i = 0 ; i < num_feat ; i ++ )
    {
      if ( pFeatStatus[i] && pFeatScale[i] >= nLevels-MTD_TRAIN_SCALES )
      {
        float x_prev = pFeat_prev[i].x * pow2[pFeatScale[i]];
        float y_prev = pFeat_prev[i].y * pow2[pFeatScale[i]];
        if ( abs(x1 - x_prev) < 0.7*width_half &&
          abs(y1 - y_prev) < 0.7*height_half )
        {
          dx += ((pFeat_curr[i].x-pFeat_prev[i].x) * pow2[pFeatScale[i]]);
          dy += ((pFeat_curr[i].y-pFeat_prev[i].y) * pow2[pFeatScale[i]]);
          good_count += 1;
        }
      }
    }
    // train mtd
    if ( good_count > 0 )
    {
      x1 += (int)(dx/good_count);
      y1 += (int)(dy/good_count);
      track_x[0] = x1;
      track_y[0] = y1;
      trackMTD->Learn( ppPyramid_curr, nLevels, num_feat, pFeat_curr, pFeatScale, track_x[0], track_y[0], track_l[0], track_s[0], track_r[0] );

      for ( int i = 0 ; i < trackMTD->mtd[0]->btd_filt[0]->num_histogram ; i ++ )
      {
        cvCircle( pImageFrame,
             cvPoint(x1+trackMTD->mtd[0]->btd_filt[0]->hist_x[i]*pow2[MTD_TRAIN_SCALES], y1+trackMTD->mtd[0]->btd_filt[0]->hist_y[i]*pow2[MTD_TRAIN_SCALES]),
             trackMTD->mtd[0]->btd_filt[0]->sample_step/2*pow2[MTD_TRAIN_SCALES]-1,
             CV_RGB(trackMTD->mtd[0]->btd[1]->weight[i]*255,trackMTD->mtd[0]->btd[0]->weight[i]*255,trackMTD->mtd[0]->btd[0]->weight[i]*255),
             -1,1,0);
      }
    }
  }
}

void TST_TEST(IplImage* pImageFrame, IplImage* pImageGray){
  num_feat = 0;
  for ( int i = nLevels - 1 ; i >= 0 ; i -- )
  {
    if ( num_feat >= max_num_feat ) break;

          vector<KeyPoint> keypoints;
          Mat * temp = new Mat(ppPyramid_prev[i]);
          FAST( *temp, keypoints, 10 );
          int num_new_corners = keypoints.size();
          if (num_new_corners == 0) continue;
          sort( keypoints.begin(), keypoints.end(), DescendingOrder);
    float dist_threshold = (BTD_NUM_ROWS*BTD_SAMPLE_STEP/3)*(BTD_NUM_ROWS*BTD_SAMPLE_STEP/3);
    /*
    for ( int j = 0 ; j < num_new_corners && num_feat < max_num_feat ; j ++ )
    {
      // check the distance to other corners
      bool good_feature = true;
      for ( int k = 0 ; good_feature && k < num_feat ; k ++ )
      {
        float dist = (pFeat[k].x - new_corners[j].x)*(pFeat[k].x - new_corners[j].x)+
        (pFeat[k].y - new_corners[j].y)*(pFeat[k].y - new_corners[j].y);
        if ( pFeatScale[k] == i && dist < dist_threshold )
        {
          good_feature = false;
        }
      }
      if ( !good_feature ) continue;

      // add features
      pFeat[num_feat] = new_corners[j];
      pFeatScale[num_feat] = i;
      num_feat ++;
    }*/
    if ( num_new_corners > 0 )
    {
      // from previous region
      int curr_pow = pow2[i];
      for ( int j = 0 ; j < num_new_corners ; j ++ )
      {
                  float pt_x = keypoints[j].pt.x;
                  float pt_y = keypoints[j].pt.y;
        if ( pt_x*curr_pow >= region[0] && pt_x*curr_pow <= region[2] &&
           pt_y*curr_pow >= region[1] && pt_y*curr_pow <= region[3] )
        {

          // check the distance to other corners
          bool good_feature = true;
          for ( int k = 0 ; good_feature && k < num_feat ; k ++ )
          {
            float dist = (pFeat[k].x - pt_x)*(pFeat[k].x - pt_x)+
            (pFeat[k].y - pt_y)*(pFeat[k].y - pt_y);
            if ( pFeatScale[k] == i && dist < dist_threshold )
            {
              good_feature = false;
            }
          }
          if ( good_feature )
          {
            pFeat[num_feat] = cvPoint(pt_x, pt_y);
            pFeatScale[num_feat] = i;
            num_feat ++;
          }
        }
      }
      // random selection
      for ( int j = 0 ; j < (nLevels-i)*(nLevels-i)*2 ; j ++ )
      {
        int k = rand() % num_new_corners;
        pFeat[num_feat] = cvPoint(keypoints[k].pt.x, keypoints[k].pt.y);
        pFeatScale[num_feat] = i;
        num_feat ++;
      }
    }
  }
  // template matching
  float width_half = 0;
  float height_half = 0;

  int prev_x = track_x[0];
  int prev_y = track_y[0];
  float score_threshold = MTD_SCORE_THRESHOLD;

  int num_win = 0;
  /*      num_match = trackMTD->DetectSlidingWindow( ppPyramid_curr, nLevels,
   max_num_match, track_x, track_y, track_l, track_s, track_r, track_score, score_threshold, &num_win );/**/
  num_match = trackMTD->Detect( ppPyramid_curr, nLevels, num_feat, pFeat, pFeatScale,
                 MTD_PRUNE_TRACKING, track_x[0], track_y[0], track_l[0], track_s[0], track_r[0],
                 max_num_match, track_x, track_y, track_l, track_s, track_r, track_score, score_threshold, &num_win );/**/
  // find best match
  if ( num_match > 0 )
  {
    int best_match = 0;
    int best_dist = (prev_x-track_x[0])*(prev_x-track_x[0])+(prev_y-track_y[0])*(prev_y-track_y[0]);
    for ( int i = 0 ; i < num_match ; i ++ )
    {
      width_half = trackMTD->get_mtd_width( track_l[i], track_s[i] ) / 2;
      height_half = trackMTD->get_mtd_height( track_l[i], track_s[i] ) / 2;
      cvRectangle(pImageFrame,
            cvPoint(track_x[i]-width_half,track_y[i]-height_half),
            cvPoint(track_x[i]+width_half,track_y[i]+height_half),
            CV_RGB(0,255,255), 1, 1, 0);/**/

      if ( track_score[best_match] > track_score[i] ||
        ( track_score[best_match] == track_score[i] && prev_x >= 0 &&
         (prev_x-track_x[i])*(prev_x-track_x[i])+(prev_y-track_y[i])*(prev_y-track_y[i]) < best_dist ) )
        //          if ( ( prev_x < 0 && track_score[best_match] > track_score[i] ) ||
        //             ( prev_x >= 0 && (prev_x-track_x[i])*(prev_x-track_x[i])+(prev_y-track_y[i])*(prev_y-track_y[i]) < best_dist ) )
      {
        best_dist = (prev_x-track_x[i])*(prev_x-track_x[i])+(prev_y-track_y[i])*(prev_y-track_y[i]);
        best_match = i;
      }
    }
    track_x[0] = track_x[best_match];
    track_y[0] = track_y[best_match];
    track_l[0] = track_l[best_match];
    track_s[0] = track_s[best_match];
    track_r[0] = track_r[best_match];
    track_score[0] = track_score[best_match];
    width_half  = trackMTD->get_mtd_width ( track_l[0], track_s[0] ) / 2;
    height_half = trackMTD->get_mtd_height( track_l[0], track_s[0] ) / 2;
    cvRectangle(pImageFrame,
          cvPoint(track_x[0]-width_half,track_y[0]-height_half),
          cvPoint(track_x[0]+width_half,track_y[0]+height_half),
          CV_RGB(0,255,0), 3, 1, 0);
    //trackMTD->Learn(ppPyramid_curr, nLevels, num_btd_test, pFeatTest, pFeatTestScale, track_x[0], track_y[0], track_l[0], track_s[0], track_r[0]);

    geometry_msgs::Vector3 pt;

    pt.x = track_x[0];
    pt.y = track_y[0];
    pt.z = ((float)(height_half*2*width_half*2))/((float)(pImageGray->height*pImageGray->width));
    recFr_pub_.publish(pt);
    cout<<"Published Detection at "<<track_x[0]<<" "<<track_y[0]<<" "<<((float)(height_half*2*width_half*2))/((float)(pImageGray->height*pImageGray->width))<<endl;
  }
  else
  {
    track_x[0] = -1;
    track_y[0] = -1;
    track_l[0] = -1;
    width_half = 0;
    height_half = 0;
    track_score[0] = MTD_SCORE_THRESHOLD;

    //No detection publish
    geometry_msgs::Vector3 pt;

    pt.x = 0;
    pt.y = 0;
    pt.z = 0;
    recFr_pub_.publish(pt);
  }
  region[0] = track_x[0]-width_half;
  region[1] = track_y[0]-height_half;
  region[2] = track_x[0]+width_half;
  region[3] = track_y[0]+height_half;
}
