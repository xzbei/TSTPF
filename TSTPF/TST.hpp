/*
 *  TST.h
 *  Tracker-LeeS10
 *
 *  Created by Taehee Lee on 2/16/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef _TST_H_
#define _TST_H_

#include <opencv2/opencv.hpp>

// initialize image pyramid (first time)
void init_image_pyramid( IplImage * image, IplImage ** pyramid, int nlevels );

// build image pyramid (when pyramid is already initialized)
void build_image_pyramid( IplImage * image, IplImage ** pyramid, int nlevels );

// track features using Tracking on the Selection Tree (TST) algorithm
void trackTST( IplImage ** imPyr1, IplImage ** imPyr2, CvPoint2D32f * point1, CvPoint2D32f * point2, int * scale, int num_point, char * status );

#endif // _TST_H_
