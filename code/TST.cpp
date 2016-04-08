/*
 *  TST.cpp
 *  Tracker-LeeS10
 *
 *  Created by Taehee Lee on 2/16/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "TST.h"

#include <cv.h>
#include <highgui.h>
 #include <stdio.h>
 using namespace std;
 using namespace cv;

void init_image_pyramid( IplImage * image, IplImage ** pyramid, int nlevels )
{
	pyramid[0] = cvCloneImage( image );
	int w = cvGetSize(image).width;
	int h = cvGetSize(image).height;
	for ( int i = 1 ; i < nlevels ; i ++ )
	{
		w = cvRound(w/2.0);
		h = cvRound(h/2.0);
		pyramid[i] = cvCreateImage( cvSize(w,h), image->depth, image->nChannels );
		cvPyrDown( pyramid[i-1], pyramid[i] );
	}
}

void build_image_pyramid( IplImage * image, IplImage ** pyramid, int nlevels )
{
	/*
	cout<<"Pyramiding with levels "<<pyramid[0]->nChannels<<" "<<image->nChannels<<" "<<pyramid[0]->depth<<endl;

	cout<<"Showing image.."<<endl;
	Mat * temp = new Mat(image);
  	imshow("DEBUG", *temp);
  	waitKey(500);
  	cout<<"showing py"<<endl;
  	temp = new Mat(pyramid[0]);
  	cout<<"showing py 2 "<<endl;
  	imshow("DEBUG", *temp);
  	cout<<"showing py 3 "<<endl;
  	waitKey(500);
  	cout<<"showing py 4 "<<endl;*/

	cvCopy( image, pyramid[0] );
	for ( int i = 1 ; i < nlevels ; i ++ )
	{
		cvPyrDown( pyramid[i-1], pyramid[i] );
	}
}

void trackTST( IplImage ** imPyr1, IplImage ** imPyr2, CvPoint2D32f * point1, CvPoint2D32f * point2, int * scale, int num_point, char * status )
{
	// parameters
	int win_size = 7;
	int win_half = win_size/2;
	int win_track_size = 11;

	// we assume that the feature points are sorted in descending order of scale. (high scale value = coarse level in pyramid)

	// variables
	int level_max = scale[0];
	int level_min = scale[num_point-1];
	int index = 0;
	int index_level = 0;

	// make a table for power of 2's
	float * pow2 = (float *)malloc( sizeof(float) * (level_max+1) );
	pow2[0] = 1.0f;
	for ( int i = 1 ; i <= level_max ; i ++ )
	{
		pow2[i] = 2 * pow2[i-1];
	}

	// variables for NCC (3x3) verification
	float ncc;
	CvMat nccMat = cvMat(1, 1, CV_32FC1, &ncc);
	IplImage * nccWin1 = cvCreateImage( cvSize(3,3), IPL_DEPTH_8U, 1 );
	IplImage * nccWin2 = cvCreateImage( cvSize(3,3), IPL_DEPTH_8U, 1 );

	// process from coarse level
	for ( int level = level_max ; level >= level_min ; level -- )
	{
		// keep the first index of the current level
		index_level = index;

		// process the features in the current level
		while ( index < num_point && scale[index] == level )
		{
			// find a parent from upper levels
			bool found_parent = false;
			float best_dist = win_half*win_half;
			int best_index = -1;
			for ( int i = index_level-1 ; i >= 0 ; i -- )
			{
				// check for the features with valid status
				if ( !status[i] ) continue;

				// compute the distance in the parent's level
				float dx = point1[index].x/pow2[scale[i]-level] - point1[i].x;
				float dy = point1[index].y/pow2[scale[i]-level] - point1[i].y;
				float dist = dx*dx+dy*dy;
				// choose the closest parent
				if ( dist < best_dist )
				{
					best_dist = dist;
					best_index = i;
					found_parent = true;
				}
				// check if we looked at all features in the level
				if ( best_index >= 0 && scale[best_index] < scale[i] )
				{
					break;
				}
			}
			// propagate the parent's motion//?
			if ( found_parent )
			{
				point2[index].x = point1[index].x + (point2[best_index].x - point1[best_index].x) * pow2[scale[best_index]-level];
				point2[index].y = point1[index].y + (point2[best_index].y - point1[best_index].y) * pow2[scale[best_index]-level];
			}
			else
			{
				// if there is no parent,
				// take the coordinate as it is
				point2[index] = point1[index];
			}

			// increase index
			index ++;
 		}

		// perform a single-level lucas-kanade tracking
		int num_point_level = index - index_level;
		if ( num_point_level > 0 )
		{
			cvCalcOpticalFlowPyrLK(
								   imPyr1[level],
								   imPyr2[level],
								   0,
								   0,
								   &(point1[index_level]),
								   &(point2[index_level]),
								   num_point_level,
								   cvSize(win_track_size,win_track_size),
								   0,
								   &(status[index_level]),
								   0,
								   cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,25,0.002),
								   CV_LKFLOW_INITIAL_GUESSES );
			// check the image boundaries
			int width = imPyr1[level]->width;
			int height= imPyr1[level]->height;
			int x1, y1, x2, y2;
			for ( int j = index_level ; j < index_level+num_point_level ; j ++ )
			{
				// check the image boundaries
				if ( point2[j].x < win_half || point2[j].x >= width-win_half ||
					point2[j].y < win_half || point2[j].y >= height-win_half )
				{
					status[j] = 0;
					continue;
				}
			}
		}
	}

	// free memory
	free( pow2 );
	cvReleaseImage( &nccWin1 );
	cvReleaseImage( &nccWin2 );
}
