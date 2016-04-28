//
//  observation.h
//  TSTPF
//
//  Created by BeiXinzhu on 4/7/16.
//  Copyright © 2016 BeiXinzhu. All rights reserved.
//

#ifndef observation_h
#define observation_h

#include <stdio.h>
//#include "BTD.hpp"
#include "defs.h"


#define NH 10
#define NS 10
#define NV 10

/* max HSV values */
#define H_MAX 360.0
#define S_MAX 1.0
#define V_MAX 1.0

/* low thresholds on saturation and value for histogramming */
#define S_THRESH 0.1
#define V_THRESH 0.2

/* distribution parameter */
#define LAMBDA1 5
#define LAMBDA2 0.5

//#ifdef _cplusplus
//extern "C"{
//#endif
#ifdef __cplusplus
extern "C" {
#endif

    typedef struct histogram {
        float histo[NH*NS + NV];   /**< histogram array */
        int n;                     /**< length of histogram array */
    } histogram;

    IplImage* bgr2hsv( IplImage* img );
    int histo_bin( float h, float s, float v );
    histogram* calc_histogram( IplImage** imgs, int n );
    void normalize_histogram( histogram* histo );
    float histo_dist_sq( histogram* h1, histogram* h2 );
    float likelihood( IplImage* img, int r, int c, int w, int h, histogram* ref_histo );
        IplImage* likelihood_image( IplImage* img, int w, int h, histogram* ref_histo);
    int export_histogram( histogram* histo, char* filename );
    // int getregions(IplImage* frame, CvRect** regions);
    histogram** compute_ref_histos( IplImage* frame, CvRect* regions, int n );

#ifdef __cplusplus
}
#endif

#endif /* observation_h */
