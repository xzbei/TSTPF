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
using namespace GlobalConst;
using namespace GlobalVar;

void PF_train(IplImage* frame, IplImage* hsv_frame);
particle* PF_init(IplImage* pImageFrame, IplImage* hsv_frame);
void PF_test(IplImage* frame, IplImage* hsv_frame);

#endif /* PFTrack_hpp */
