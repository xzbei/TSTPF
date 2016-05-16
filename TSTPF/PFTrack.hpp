//
//  PFTrack.hpp
//  TSTPF
//
//  Created by BeiXinzhu on 4/7/16.
//  Copyright © 2016 BeiXinzhu. All rights reserved.
//

#ifndef PFTrack_hpp
#define PFTrack_hpp

#include <stdio.h>
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
#include "unistd.h"
#include "opencv/cv.h"
#include "globaldata.hpp"

using namespace std;
using namespace cv;
using namespace TST;
using namespace TSTPF;
using namespace PF;

void PF_train(IplImage* frame, IplImage* hsv_frame);
particle* PF_init(IplImage* pImageFrame, IplImage* hsv_frame);
void PF_test(IplImage* frame, IplImage* hsv_frame, IplImage* framegrey);
float likelihood2( IplImage* img, int r, int c, int w, int h, IplImage ** pyramid, MTD ** mtd ,float threshold);
float likelihood1( IplImage* img, int r, int c, int w, int h, histogram* ref_histo, bool motion_like);
float motion_likelihood(int x, int y, int w,int h);
void display_boundingbox(IplImage* img,particle p, CvScalar color, int a, int b ,int c);


#endif /* PFTrack_hpp */
