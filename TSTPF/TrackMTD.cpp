/*
 *  TrackMTD.cpp
 *  BTD
 *
 *  Created by Taehee Lee on 11/3/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include "TrackMTD.hpp"

TrackMTD::TrackMTD()
{
    num_max_mtd = 10;
    num_templates = 2;
}

//TrackMTD::~TrackMTD()
//{
//}

void TrackMTD::Initialize( int num_row, int num_col, int num_bin, int num_step, int _num_scale, int _num_anchors, int _width, int _height )
{
    num_scale = _num_scale;
    num_anchors = _num_anchors;
    width = _width;
    height = _height;

    // init BTD filters for intensity and gradient orientations
    btd_filt_i = (BTD_Filter **)malloc( sizeof(BTD_Filter *) * num_scale );
    btd_filt_g = (BTD_Filter **)malloc( sizeof(BTD_Filter *) * num_scale );
    for ( int i = 0 ; i < num_scale ; i ++ )
    {
        float sample_step = num_step*(1+i/(float)num_scale);
        btd_filt_i[i] = BTD_Filter_create( num_row, num_col, num_bin, sample_step, BTD_METHOD_INTENSITY );
        btd_filt_g[i] = BTD_Filter_create( num_row, num_col, num_bin, sample_step, BTD_METHOD_GRADIENT );
    }

    // init MTD - mixture of template descriptors
    mtd_filt = (BTD_Filter **)malloc( sizeof(BTD_Filter *) * num_templates );
    mtd_filt[0] = btd_filt_i[0];
    mtd_filt[1] = btd_filt_g[0];

    mtd = (MTD **)malloc( sizeof(MTD *) * num_max_mtd );
    for ( int i = 0 ; i < num_max_mtd ; i ++ )
    {
        mtd[i] = MTD_create( num_templates, num_anchors, mtd_filt );
    }
    num_mtd = 0;

    // for matching
    visited = (int *)malloc( sizeof(int) * width * height );
    memset( visited, 0, sizeof(int) * width * height );


    pow2 = (int *)malloc( sizeof(int) * 10 );
    pow2[0] = 1;
    for ( int i = 1 ; i < 10 ; i ++ )
        pow2[i] = pow2[i-1] * 2;

    // for kalman filter
    kalman = cvCreateKalman( 4, 2, 0 );
    kalman_state = cvCreateMat( 4, 1, CV_32FC1 );
    kalman_process_noise = cvCreateMat( 4, 1, CV_32FC1 );
    kalman_measurement = cvCreateMat( 2, 1, CV_32FC1 );
    float A[] = { 1, 0, 0.1, 0, \
        0, 1, 0, 0.1, \
        0, 0, 1, 0, \
        0, 0, 0, 1 };
    memcpy( kalman->transition_matrix->data.fl, A, sizeof(A) );
    //	cvSetIdentity( kalman->transition_matrix, cvRealScalar(1));
    cvSetIdentity( kalman->measurement_matrix, cvRealScalar(1) );
    cvSetIdentity( kalman->process_noise_cov, cvRealScalar(1) );
    cvSetIdentity( kalman->measurement_noise_cov, cvRealScalar(1) );
    cvSetIdentity( kalman->error_cov_post, cvRealScalar(1));

    cvRandInit( &rng, 0, 1, -1, CV_RAND_UNI );
    cvZero( kalman_measurement );
    cvZero( kalman_state );
    kalman_state->data.fl[0] = -1;

    rng.disttype = CV_RAND_NORMAL;
}

float TrackMTD::get_mtd_width( int level, int scale )
{
    return ( pow2[level] * btd_filt_i[scale]->num_col * btd_filt_i[scale]->sample_step );
}

float TrackMTD::get_mtd_height( int level, int scale )
{
    return ( pow2[level] * btd_filt_i[scale]->num_row * btd_filt_i[scale]->sample_step );
}


void TrackMTD::Learn( IplImage ** pyramid, int num_level, int num_feat, CvPoint2D32f * feat, int * feat_scale, int x, int y, int level, int scale, int angle )
{
    // learn feature distribution
    float thresh_width  = 0.8 * get_mtd_width ( level, scale ) / 2;
    float thresh_height = 0.8 * get_mtd_height( level, scale ) / 2;
    for ( int i = 0 ; i < num_feat ; i ++ )
    {
        float feat_x = feat[i].x * pow2[feat_scale[i]];
        float feat_y = feat[i].y * pow2[feat_scale[i]];
        if ( abs(x - feat_x) < thresh_width && abs(y - feat_y) < thresh_height )
        {
            MTD_update_feat_dist( mtd[0], x, y, feat_x, feat_y, pow2[level] );
        }
    }
    MTD_select_anchors( mtd[0] );

    // learn templates
    for ( int i = 0 ; i <= level ; i ++ )
    {
        if ( i < num_level )
        {
            MTD_learn( mtd[0],
                      (unsigned char *)pyramid[i]->imageData,
                      pyramid[i]->widthStep,
                      pyramid[i]->width,
                      pyramid[i]->height,
                      x/pow2[i], y/pow2[i], pow2[level-i], angle );
        }
    }
}

void TrackMTD::Learn( IplImage ** pyramid, int num_level, int num_feat, CvPoint * feat, int * feat_scale, int x, int y, int level, int scale, int angle )
{
    // learn feature distribution
    float thresh_width  = 0.7 * get_mtd_width ( level, scale ) / 2;
    float thresh_height = 0.7 * get_mtd_height( level, scale ) / 2;
    for ( int i = 0 ; i < num_feat ; i ++ )
    {
        float feat_x = feat[i].x * pow2[feat_scale[i]];
        float feat_y = feat[i].y * pow2[feat_scale[i]];
        if ( abs(x - feat_x) < thresh_width && abs(y - feat_y) < thresh_height )
        {
            MTD_update_feat_dist( mtd[0], x, y, feat_x, feat_y, pow2[level] );
        }
    }
    MTD_select_anchors( mtd[0] );

    // learn templates
    for ( int i = 0 ; i <= level ; i ++ )
    {
        if ( i < num_level )
        {
            MTD_learn( mtd[0],
                      (unsigned char *)pyramid[i]->imageData,
                      pyramid[i]->widthStep,
                      pyramid[i]->width,
                      pyramid[i]->height,
                      x/pow2[i], y/pow2[i], pow2[level-i], angle );
        }
    }
}

int TrackMTD::Detect( IplImage ** pyramid, int num_level, int num_feat, CvPoint * feat, int * feat_scale, bool prune_tracking, int prev_x, int prev_y, int prev_level, int prev_scale, int prev_angle, int num_max, int * x, int * y, int * level, int * scale, int * angle, float * score, float threshold, int * num_win )
{
    int num_detected = 0;
    int num_tested = 0;

    float win_half = 0;
    unsigned char * curr_image = 0;
    int curr_stride, curr_width, curr_height;
    int curr_pow2 = 0;
    for ( int i_level = 0 ; i_level < num_level ; i_level ++ )
    {
        // check with the previous level
        if ( prune_tracking && prev_level >= 0 && abs(prev_level-i_level) > 1 ) continue;

        curr_image  = (unsigned char *)(pyramid[i_level]->imageData);
        curr_stride = pyramid[i_level]->widthStep;
        curr_width  = pyramid[i_level]->width;
        curr_height = pyramid[i_level]->height;
        curr_pow2   = pow2[i_level];

        for ( int i_scale = 0 ; i_scale < num_scale ; i_scale ++ )
        {
            win_half = get_mtd_width( 0, i_scale ) / 2;
            mtd[0]->btd_filt[0] = btd_filt_i[i_scale];
            mtd[0]->btd_filt[1] = btd_filt_g[i_scale];

            //			memset( visited, 0, sizeof(int) * curr_width * curr_height );

            for ( int i = 0 ; i < num_feat ; i ++ )
            {
                int feat_x = (feat[i].x)*pow2[feat_scale[i]];
                int feat_y = (feat[i].y)*pow2[feat_scale[i]];

                for ( int j = 0 ; j < mtd[0]->num_anchor ; j ++ )
                {
                    // get the center
                    float x_ = feat_x/curr_pow2 - mtd[0]->btd_filt[0]->hist_x[ mtd[0]->anchor[j] ];
                    float y_ = feat_y/curr_pow2 - mtd[0]->btd_filt[0]->hist_y[ mtd[0]->anchor[j] ];

                    // check with the previous position
                    if ( prune_tracking && prev_x >= 0 && (prev_x-x_*curr_pow2)*(prev_x-x_*curr_pow2)+(prev_y-y_*curr_pow2)*(prev_y-y_*curr_pow2) > 50*50 ) continue;
                    // check with the image boundary
                    if ( x_ < win_half || x_ >= curr_width -win_half ||
                        y_ < win_half || y_ >= curr_height-win_half )
                        continue;
                    // check with the visited position
                    //					if ( visited[((int)y_)*curr_width+(int)x_] ) continue;
                    //					visited[((int)y_)*curr_width+(int)x_] = 1;

                    num_tested ++;
                    float match_score = MTD_test( mtd[0], curr_image, curr_stride, curr_width, curr_height, x_, y_, 0, threshold );
                    if ( match_score < threshold && num_detected < num_max )
                    {
                        x_ *= curr_pow2;
                        y_ *= curr_pow2;

                        // check locally
                        bool good_to_add = true;
                        for ( int i = 0 ; i < num_detected ; i ++ )
                        {
                            // if there exists an overlapping region, take the better one
                            int win = get_mtd_width( level[i], scale[i] );
                            if ( (x[i]-x_)*(x[i]-x_)+(y[i]-y_)*(y[i]-y_) < 0.25*win*win )
                            {
                                if ( score[i] > match_score )
                                {
                                    x[i] = x_;
                                    y[i] = y_;
                                    level[i] = i_level;
                                    scale[i] = i_scale;
                                    score[i] = match_score;
                                }
                                good_to_add = false;
                                break;
                            }
                        }

                        // if new, add this one
                        if ( good_to_add )
                        {
                            x[num_detected] = x_;
                            y[num_detected] = y_;
                            level[num_detected] = i_level;
                            scale[num_detected] = i_scale;
                            score[num_detected] = match_score;
                            num_detected ++;
                        }

                        if ( prune_tracking ) threshold = match_score;
                    }
                }
            }
        }
    }
    mtd[0]->btd_filt[0] = btd_filt_i[0];
    mtd[0]->btd_filt[1] = btd_filt_g[0];

    // post processing

    if ( num_win )
    {
        *num_win = num_tested;
    }

    return num_detected;
}

int TrackMTD::DetectSlidingWindow( IplImage ** pyramid, int num_level, int num_max, int * x, int * y, int * level, int * scale, int * angle, float * score, float threshold, int * num_win )
{
    int num_detected = 0;
    int num_tested = 0;

    float win_half = 0;
    unsigned char * curr_image = 0;
    int curr_stride, curr_width, curr_height;
    int curr_pow2 = 0;
    for ( int i_level = 0 ; i_level < num_level ; i_level ++ )
    {
        curr_image  = (unsigned char *)(pyramid[i_level]->imageData);
        curr_stride = pyramid[i_level]->widthStep;
        curr_width  = pyramid[i_level]->width;
        curr_height = pyramid[i_level]->height;
        curr_pow2   = pow2[i_level];

        for ( int i_scale = 0 ; i_scale < num_scale ; i_scale ++ )
        {
            win_half = get_mtd_width( 0, i_scale ) / 2;
            mtd[0]->btd_filt[0] = btd_filt_i[i_scale];
            mtd[0]->btd_filt[1] = btd_filt_g[i_scale];

            for ( int win_x = win_half ; win_x < curr_width - win_half ; win_x += 0.4*win_half )
            {
                for ( int win_y = win_half ; win_y < curr_height - win_half ; win_y += 0.4*win_half )
                {
                    num_tested ++;
                    float match_score = MTD_test( mtd[0], curr_image, curr_stride, curr_width, curr_height, win_x, win_y, 0, threshold );
                    if ( match_score < threshold && num_detected < num_max )
                    {
                        int x_ = win_x * curr_pow2;
                        int y_ = win_y * curr_pow2;

                        // check locally
                        bool good_to_add = true;
                        for ( int i = 0 ; i < num_detected ; i ++ )
                        {
                            // if there exists an overlapping region, take the better one
                            int win = get_mtd_width( level[i], scale[i] );
                            if ( (x[i]-x_)*(x[i]-x_)+(y[i]-y_)*(y[i]-y_) < 0.25*win*win )
                            {
                                if ( score[i] > match_score )
                                {
                                    x[i] = x_;
                                    y[i] = y_;
                                    level[i] = i_level;
                                    scale[i] = i_scale;
                                    score[i] = match_score;
                                }
                                good_to_add = false;
                                break;
                            }
                        }

                        // if new, add this one
                        if ( good_to_add )
                        {
                            x[num_detected] = x_;
                            y[num_detected] = y_;
                            level[num_detected] = i_level;
                            scale[num_detected] = i_scale;
                            score[num_detected] = match_score;
                            num_detected ++;
                        }
                    }
                }
            }
        }
    }
    mtd[0]->btd_filt[0] = btd_filt_i[0];
    mtd[0]->btd_filt[1] = btd_filt_g[0];

    // post processing

    if ( num_win )
    {
        *num_win = num_tested;
    }

    return num_detected;
}

void TrackMTD::Track( int x, int y, int s, float *x_, float *y_, float *s_ )
{
    // check reset state
    if ( kalman_state->data.fl[0] < 0 && x > 0 )
    {
        kalman_state->data.fl[0] = x;
        kalman_state->data.fl[1] = y;
        kalman_state->data.fl[2] = 0;
        kalman_state->data.fl[3] = 0;
    }

    for ( int i = 0 ; i < 4 ; i ++ )
        printf("%.2f ", kalman_state->data.fl[i]);
    printf("\n");

    // predict
    cvKalmanPredict( kalman, 0 );

    // measurement
    kalman_measurement->data.fl[0] = x;
    kalman_measurement->data.fl[1] = y;

    // update
    cvKalmanCorrect( kalman, kalman_measurement );
    rng.disttype = CV_RAND_NORMAL;
    cvRandSetRange( &rng, 0, sqrt(kalman->process_noise_cov->data.fl[0]), 0 );
    cvRand( &rng, kalman_process_noise );
    cvMatMulAdd( kalman->transition_matrix, kalman_state, kalman_process_noise, kalman_state );

    // return state
    *x_ = (int)(kalman_state->data.fl[0]);
    *y_ = (int)(kalman_state->data.fl[1]);
    *s_ = s;//(int)(kalman_state->data.fl[2]);
}
