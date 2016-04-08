/*
 *  BTD.h
 *  BTD
 *
 *  Created by Taehee Lee on 7/18/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef _BTD_H_
#define _BTD_H_

#define BTD_METHOD_INTENSITY	0
#define BTD_METHOD_GRADIENT		1

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

BTD_Filter * BTD_Filter_create( int num_row, int num_col, int num_bin, float sample_step, int method );
void BTD_Filter_free( BTD_Filter ** btd_filt );

BTD * BTD_create( BTD_Filter * btd_filt );
void BTD_clear( BTD_Filter * btd_filt, BTD * btd );
void BTD_free( BTD_Filter * btd_filt, BTD ** btd );

void BTD_update( BTD_Filter * btd_filt, unsigned char * image, int stride, int width, int height, int x, int y, BTD * btd, float angle0 );
void BTD_update_intensity( BTD_Filter * btd_filt, unsigned char * image, int stride, int width, int height, int x, int y, BTD * btd, float angle0 );
void BTD_update_gradient( BTD_Filter * btd_filt, unsigned char * image, int stride, int width, int height, int x, int y, BTD * btd, float angle0 );

void BTD_learn( BTD_Filter * btd_filt, unsigned char * image, int stride, int width, int height, int x, int y, int scale, float angle, BTD * btd );
void BTD_learn_intensity( BTD_Filter * btd_filt, unsigned char * image, int stride, int width, int height, int x, int y, int scale, float angle, BTD * btd );
void BTD_learn_gradient( BTD_Filter * btd_filt, unsigned char * image, int stride, int width, int height, int x, int y, int scale, float angle, BTD * btd );

float BTD_compare( BTD_Filter * btd_filt, BTD * btd_sample, BTD * btd_reference );
float BTD_quality( BTD_Filter * btd_filt, BTD * btd );

void init_bit_count();
void BTD_make_compact_reference( BTD_Filter * btd_filt, BTD * btd );
void BTD_make_compact_sample( BTD_Filter * btd_filt, BTD * btd );
int BTD_compare_compact( BTD_Filter * btd_filt, BTD * btd_sample, BTD * btd_reference );

float BTD_test( BTD_Filter * btd_filt, BTD * btd_reference, unsigned char * image, int stride, int width, int height, int x, int y, float angle );
float BTD_test_intensity( BTD_Filter * btd_filt, BTD * btd_reference, unsigned char * image, int stride, int width, int height, int x, int y, float angle );
float BTD_test_gradient( BTD_Filter * btd_filt, BTD * btd_reference, unsigned char * image, int stride, int width, int height, int x, int y, float angle );

float get_orientation( unsigned char * image, int stride, int x, int y );
float get_sample( unsigned char * image, int stride, int width, int height, float x, float y );

// mixture of template descriptors
MTD * MTD_create( int num_template, int num_anchor, BTD_Filter ** btd_filt );
void MTD_free( MTD ** mtd );
void MTD_learn( MTD * mtd, unsigned char * image, int stride, int width, int height, int x, int y, int scale, float angle );
void MTD_update( MTD * mtd, unsigned char * image, int stride, int width, int height, int x, int y, float angle0 );
void MTD_sample( MTD * mtd, unsigned char * image, int stride, int width, int height, int x, int y, float angle0 );
float MTD_test( MTD * mtd, unsigned char * image, int stride, int width, int height, int x, int y, float angle, float threshold );
void MTD_update_feat_dist( MTD * mtd, int x, int y );
void MTD_update_feat_dist( MTD * mtd, int cx, int cy, int x, int y, int scale );
void MTD_select_anchors( MTD * mtd );
void MTD_make_compact_reference( MTD * mtd );
void MTD_make_compact_sample( MTD * mtd );
float MTD_compare( MTD * mtd_sample, MTD * mtd_reference );
float MTD_compare_compact( MTD * mtd_sample, MTD * mtd_reference );

#endif // _BTD_H_
