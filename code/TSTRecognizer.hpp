/*
 *  BTD.h
 *  BTD
 *
 *  Created by Taehee Lee on 7/18/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef _TSTRECOGINZER_H_
#define _TSTRECOGINZER_H_

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
#include "globaldata.hpp"

using namespace cv;
using namespace std;

bool DescendingOrder(cv::KeyPoint pt1, cv::KeyPoint pt2);
void initializeTracker(IplImage* pImageFrame, IplImage* pImageGray);
void TST_prep(IplImage* pImageFrame, IplImage* pImageGray);
void TST_RESET(IplImage* pImageFrame, IplImage* pImageGray);
void TST_BEGIN(IplImage* pImageFrame, IplImage* pImageGray);
void TST_TRAIN(IplImage* pImageFrame, IplImage* pImageGray);
void TST_TEST(IplImage* pImageFrame, IplImage* pImageGray);

#endif // _TSTRECOGINZER_H_
