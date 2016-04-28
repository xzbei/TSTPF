//
//  def.h
//  TSTPF
//
//  Created by BeiXinzhu on 4/7/16.
//  Copyright © 2016 BeiXinzhu. All rights reserved.
//

#ifndef def_h
#define def_h

#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
//#include "unistd.h"

/* From OpenCV library */
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"

/******************************* Defs and macros *****************************/

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef MIN
#define MIN(x,y) ( ( x < y )? x : y )
#endif
#ifndef MAX
#define MAX(x,y) ( ( x > y )? x : y )
#endif
#ifndef ABS
#define ABS(x) ( ( x < 0 )? -x : x )
#endif


typedef struct {
    int num_row;		// number of rows
    int num_col;		// number of columns
    float sample_step;	// sample step per histogram
    
    int num_histogram;	// number of histograms
    int num_bin;		// number of bins in a histogram
    float * hist_x;		// x coordinates of histograms
    float * hist_y;		// y coordinates of histograms
    
    int method;			// method of BTD (0 = intensity, 1 = gradient)
    
    int * samples;		// buffer for intensity samples
    float * angle;		// buffer for gradient angles
    float * mag;		// buffer for gradient magnitudes
} BTD_Filter;

typedef struct {
    float num_sample;	// number of samples in the histogram
    float * value;		// histogram bin counts
    float * mean;		// means of samples in each bin
} Histogram;

// Best Template Descriptor
typedef struct {
    Histogram *		hist;		// histograms per sampling region
    unsigned char * compact;	// compact representation
    
    float			sum_weight;	// sum of weights
    float *			weight;		// weights of sampling regions
} BTD;

// Mixture of Template Descriptors
typedef struct {
    int		num_template;		// number of templates
    BTD_Filter ** btd_filt;		// BTD filters
    BTD **	btd;				// template descriptors
    float *	weight;				// weights of templates
    
    int		num_anchor;			// number of anchor points
    int *	anchor;				// anchor points
    int *	feat_dist;			// feature points distribution
} MTD;

/********************************** Structures *******************************/


#endif /* def_h */

