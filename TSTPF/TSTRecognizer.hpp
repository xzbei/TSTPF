//
//  TSTRecognizer.hpp
//  TSTPF
//
//  Created by BeiXinzhu on 4/5/16.
//  Copyright © 2016 BeiXinzhu. All rights reserved.
//

#ifndef TSTRecognizer_hpp
#define TSTRecognizer_hpp

#include <stdio.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <algorithm>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv/cv.h"

#include "TST.hpp"
#include "TrackMTD.hpp"
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

#endif /* TSTRecognizer_hpp */
