/*
 *  TrackMTD.h
 *  BTD
 *
 *  Created by Taehee Lee on 11/3/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef _TRACK_MTD_H_
#define _TRACK_MTD_H_

#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>
#include "BTD.h"

class TrackMTD
{
public:
	TrackMTD();
	~TrackMTD();

	void Initialize( int num_row, int num_col, int num_bin, int num_step, int _num_scale, int _num_anchors, int _width, int _height );
	void Terminate();
	
	void Learn( IplImage ** pyramid, int num_level, int num_feat, CvPoint2D32f * feat, int * feat_scale, int x, int y, int level, int scale, int angle );
	void Learn( IplImage ** pyramid, int num_level, int num_feat, CvPoint * feat, int * feat_scale, int x, int y, int level, int scale, int angle );
	int Detect( IplImage ** pyramid, int num_level, int num_feat, CvPoint * feat, int * feat_scale, bool prune_tracking, int prev_x, int prev_y, int prev_level, int prev_scale, int prev_angle, int num_max, int * x, int * y, int * level, int * scale, int * angle, float * score, float threshold, int * num_win = 0 );
	int DetectSlidingWindow( IplImage ** pyramid, int num_level, int num_max, int * x, int * y, int * level, int * scale, int * angle, float * score, float threshold, int * num_win = 0 );
	void Track( int x, int y, int s, float *x_, float *y_, float *s_ );
	
	void Draw();
	
	float get_mtd_width( int level, int scale );
	float get_mtd_height( int level, int scale );

public:
	int				num_max_mtd;
	int				num_mtd;
	MTD **			mtd;
	BTD_Filter **	mtd_filt;
	
	int				num_templates;
	int				num_anchors;
	
	int				num_scale;
	BTD_Filter **	btd_filt_i;
	BTD_Filter **	btd_filt_g;

	
	// for matching
	int				width;
	int				height;
	int *			visited;
	
	int	*			pow2;
	
	// for kalman filter
	CvKalman *		kalman;
	CvMat *			kalman_state;
	CvMat *			kalman_process_noise;
	CvMat *			kalman_measurement;
	CvRandState		rng;
};


#endif // _TRACK_MTD_H_
