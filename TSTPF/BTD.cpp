/*
 *  BTD.cpp
 *  BTD
 *
 *  Created by Taehee Lee on 7/18/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "BTD.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "globaldata.hpp"

using namespace cv;
using namespace TST;

static int * bit_count = 0;
void init_bit_count()
{
    if ( !bit_count )
    {
        bit_count = (int *)malloc( sizeof(int) * 256 );
        for ( int i = 0 ; i < 256 ; i ++ )
        {
            int n = i;
            int count = 0;
            while ( n )
            {
                count += ( n & 0x01 );
                n >>= 1;
            }
            bit_count[i] = count;
        }
    }
}

int * atan2_table = 0;
int atan2_table_size = 101;
float atan2_table_size2 = 50;
void init_atan2_table()
{
    if ( !atan2_table )
    {
        atan2_table = (int *)malloc( sizeof(int) * atan2_table_size );
        for ( int i = 0 ; i < atan2_table_size ; i ++ )
        {
            float x = (i - atan2_table_size2)/atan2_table_size2;
            float y = sqrt( 1 - x*x );
            atan2_table[i] = atan2( y, x )*180/M_PI;
        }
    }
}


// ASSUME: max of num_bin = 8 (because the bins are represented as a byte)
BTD_Filter * BTD_Filter_create( int num_row, int num_col, int num_bin, float sample_step, int method )
{
    BTD_Filter * btd_filt = (BTD_Filter *)malloc( sizeof(BTD_Filter) );

    btd_filt->num_row = num_row;
    btd_filt->num_col = num_col;

    int num_histogram = num_row * num_col;
    btd_filt->num_histogram = num_histogram;
    btd_filt->num_bin = num_bin;
    btd_filt->hist_x = (float *)malloc( sizeof(float) * num_histogram );
    btd_filt->hist_y = (float *)malloc( sizeof(float) * num_histogram );
    btd_filt->method = method;
    btd_filt->sample_step = sample_step;

    // make histogram coordinates
    int row = num_row;
    int col = num_col;
    int even = (row%2 == 0);
    int center_row = row/2;
    int center_col = col/2;
    int k = 0;
    for ( int i = 0 ; i < row ; i ++ )
    {
        for ( int j = 0 ; j < col ; j ++ )
        {
            btd_filt->hist_x[k] = sample_step*(j-center_col)+even;
            btd_filt->hist_y[k] = sample_step*(i-center_row)+even;
            k ++;
        }
    }

    // variables
    btd_filt->samples = (int *)malloc( sizeof(int) * num_histogram );
    btd_filt->angle = (float *)malloc( sizeof(float) * num_histogram );
    btd_filt->mag = (float *)malloc( sizeof(float) * num_histogram );

    // init bit count
    init_bit_count();
    // init atan2 table
    init_atan2_table();

    return btd_filt;
}

void BTD_Filter_free( BTD_Filter ** btd_filt )
{
    free( (*btd_filt)->hist_x );
    free( (*btd_filt)->hist_y );
    free( (*btd_filt)->samples );
    free( (*btd_filt)->angle );
    free( (*btd_filt)->mag );
    free( *btd_filt );
    *btd_filt = 0;
}

BTD * BTD_create( BTD_Filter * btd_filt )
{
    int num_histogram = btd_filt->num_histogram;
    int num_bin = btd_filt->num_bin;

    BTD * btd = (BTD *)malloc( sizeof(BTD) );

    btd->hist = (Histogram *)malloc( sizeof(Histogram) * num_histogram );
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        btd->hist[i].num_sample = 0;
        btd->hist[i].value = (float *)malloc( sizeof(float) * num_bin );
        btd->hist[i].mean = (float *)malloc( sizeof(float) * num_bin );
        memset( btd->hist[i].value, 0, sizeof(float) * num_bin );
        memset( btd->hist[i].mean, 0, sizeof(float) * num_bin );
    }
    btd->compact = (unsigned char *)malloc( sizeof(unsigned char) * num_histogram );
    memset( btd->compact, 0, sizeof(unsigned char) * num_histogram );
    btd->weight = (float *)malloc( sizeof(float) * num_histogram );
    memset( btd->weight, 0, sizeof(float) * num_histogram );
    btd->sum_weight = 0;

    return btd;
}

void BTD_free( BTD_Filter * btd_filt, BTD ** btd )
{
    for ( int i = 0 ; i < btd_filt->num_histogram ; i ++ )
    {
        free( (*btd)->hist[i].value );
        free( (*btd)->hist[i].mean );
    }
    free( (*btd)->hist );
    free( (*btd)->compact );
    free( (*btd)->weight );
    free( *btd );
    *btd = 0;
}

void BTD_clear( BTD_Filter * btd_filt, BTD * btd )
{
    int num_histogram = btd_filt->num_histogram;
    int num_bin = btd_filt->num_bin;

    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        btd->hist[i].num_sample = 0;
        memset( btd->hist[i].value, 0, sizeof(float) * num_bin );
        memset( btd->hist[i].mean, 0, sizeof(float) * num_bin );
    }
    memset( btd->compact, 0, sizeof(unsigned char) * num_histogram );
    memset( btd->weight, 0, sizeof(float) * num_histogram );
    btd->sum_weight = 0;
}

void BTD_update( BTD_Filter * btd_filt, unsigned char * image, int stride, int width, int height, int x, int y, BTD * btd, float angle0 )
{
    switch ( btd_filt->method )
    {
        case BTD_METHOD_INTENSITY:
            BTD_update_intensity( btd_filt, image, stride, width, height, x, y, btd, angle0 );
            break;
        case BTD_METHOD_GRADIENT:
            BTD_update_gradient( btd_filt, image, stride, width, height, x, y, btd, angle0 );
            break;
    }
}

void BTD_update_intensity( BTD_Filter * btd_filt, unsigned char * image, int stride, int width, int height, int x, int y, BTD * btd, float angle0 )
{
    int num_histogram = btd_filt->num_histogram;
    int num_bin = btd_filt->num_bin;

    // calculate orientation
    float patch_angle = get_orientation( image, stride, x, y ) + angle0;

    // get samples from rotated sampling coordinates
    float c = cos(patch_angle);
    float s = sin(patch_angle);
    float * dir_x = btd_filt->hist_x;
    float * dir_y = btd_filt->hist_y;
    int * samples = btd_filt->samples;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        float ix = x + ( dir_x[i] * c - dir_y[i] * s );
        float iy = y + ( dir_x[i] * s + dir_y[i] * c );

        samples[i] = get_sample( image, stride, width, height, ix, iy );
    }

    // compute mean and standard deviation of the image region
    float mean = 0.0;
    float stdv = 0.0;
    float count = 0.0;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        if ( samples[i] < 0 )
            continue;

        mean += samples[i];
        count += 1.0;
    }
    mean /= count;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        if ( samples[i] < 0 )
            continue;

        float value = samples[i] - mean;
        stdv += (value*value);
    }
    stdv = sqrt(stdv/count);

    // update the histogram
    if ( stdv < 0.01 ) // NOTE: what to do here? e.g. uniform color.
    {
        for ( int i = 0 ; i < num_histogram ; i ++ )
        {
            if ( samples[i] < 0 )
                continue;
            /*
             int bin = samples[i] * num_bin / 256;

             btd->hist[i].num_sample += 1;

             btd->hist[i].mean[bin] *= btd->hist[i].value[bin];
             btd->hist[i].mean[bin] += samples[i];
             btd->hist[i].value[bin] += 1;
             btd->hist[i].mean[bin] /= btd->hist[i].value[bin];*/
            for ( int j = 0 ; j < num_bin ; j ++ )
            {
                btd->hist[i].num_sample += 1;
                btd->hist[i].value[j] += 1;
            }
        }
    }
    else
    {
        for ( int i = 0 ; i < num_histogram ; i ++ )
        {
            if ( samples[i] < 0 )
                continue;

            // normalize the pixel intensity value with the mean and standard deviation, to [0..255]
            double value = (((samples[i] - mean)/stdv/2) + 1.0 ) * 128;
            if ( value < 0 ) value = 0;
            if ( value > 255 ) value = 255;

            // get the bin index from the value
            int bin = value * num_bin / 256;

            // update the histogram
            btd->hist[i].num_sample += 1;

            btd->hist[i].mean[bin] *= btd->hist[i].value[bin];
            btd->hist[i].mean[bin] += value;
            btd->hist[i].value[bin] += 1;
            btd->hist[i].mean[bin] /= btd->hist[i].value[bin];
        }
    }
}

void BTD_update_gradient( BTD_Filter * btd_filt, unsigned char * image, int stride, int width, int height, int x, int y, BTD * btd, float angle0 )
{
    int num_histogram = btd_filt->num_histogram;
    int num_bin = btd_filt->num_bin;

    // calculate orientation
    float patch_angle = get_orientation( image, stride, x, y ) + angle0;

    // get gradients from rotated sampling coordinates
    float c = cos(patch_angle);
    float s = sin(patch_angle);
    float * dir_x = btd_filt->hist_x;
    float * dir_y = btd_filt->hist_y;
    float * angle = btd_filt->angle;
    float * mag = btd_filt->mag;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        float ix00 = x + (  dir_x[i]    * c -  dir_y[i]    * s );
        float iy00 = y + (  dir_x[i]    * s +  dir_y[i]    * c );
        float ix10 = x + ( (dir_x[i]+1) * c -  dir_y[i]    * s );
        float iy10 = y + ( (dir_x[i]+1) * s +  dir_y[i]    * c );
        float ix01 = x + (  dir_x[i]    * c - (dir_y[i]+1) * s );
        float iy01 = y + (  dir_x[i]    * s + (dir_y[i]+1) * c );

        float sample_00 = get_sample( image, stride, width, height, ix00, iy00 );
        float sample_10 = get_sample( image, stride, width, height, ix10, iy10 );
        float sample_01 = get_sample( image, stride, width, height, ix01, iy01 );

        if ( sample_00 < 0 || sample_01 < 0 || sample_10 < 0 )
        {
            angle[i] = 0;
            mag[i] = 0;
        }
        else
        {
            float dx = sample_10 - sample_00;
            float dy = sample_01 - sample_00;
            angle[i] = atan2( dy, dx )*180/M_PI;
            while ( angle[i] < 0 ) angle[i] += 360;
            while ( angle[i] >= 360 ) angle[i] -= 360;
            mag[i] = sqrt(dx*dx+dy*dy);
        }
    }

    // update the histogram
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        // NOTE: what should this threshold of magnitude be?
        if ( mag[i] <= 2 )
        {
            btd->hist[i].num_sample += 1;
            btd->hist[i].value[0] += 1;
            /*			btd->hist[i].num_sample += (mag[i]*num_bin);
             for ( int j = 0 ; j < num_bin ; j ++ )
             {
             btd->hist[i].value[j] += mag[i];
             }*/
            continue;
        }
        //		mag[i] = 1;
        if ( mag[i] >= 10 ) mag[i] = 10;

        // get value from gradient orientation
        float value = angle[i];
        // get the bin index from the value
        int bin = value * (num_bin-1) / 360 + 1;

        // update the histogram
        btd->hist[i].num_sample += mag[i];

        btd->hist[i].mean[bin] *= btd->hist[i].value[bin];
        btd->hist[i].mean[bin] += value;
        btd->hist[i].value[bin] += mag[i];
        btd->hist[i].mean[bin] /= btd->hist[i].value[bin];
    }
}

void BTD_learn( BTD_Filter * btd_filt, unsigned char * image, int stride, int width, int height, int x, int y, int scale, float angle, BTD * btd )
{
    switch ( btd_filt->method )
    {
        case BTD_METHOD_INTENSITY:
            BTD_learn_intensity( btd_filt, image, stride, width, height, x, y, scale, angle, btd );
            break;
        case BTD_METHOD_GRADIENT:
            BTD_learn_gradient( btd_filt, image, stride, width, height, x, y, scale, angle, btd );
            break;
    }
}

void BTD_learn_intensity( BTD_Filter * btd_filt, unsigned char * image, int stride, int width, int height, int x, int y, int scale, float angle, BTD * btd )
{
    int num_histogram = btd_filt->num_histogram;
    int num_bin = btd_filt->num_bin;

    // canonize orientation
    angle += get_orientation( image, stride, x, y );

    // get mean and std for normalization
    float c = cos(angle);
    float s = sin(angle);
    float * dir_x = btd_filt->hist_x;
    float * dir_y = btd_filt->hist_y;
    int * samples = btd_filt->samples;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        float ix = x + scale * ( dir_x[i] * c - dir_y[i] * s );
        float iy = y + scale * ( dir_x[i] * s + dir_y[i] * c );

        samples[i] = get_sample( image, stride, width, height, ix, iy );
    }
    float mean = 0.0;
    float stdv = 0.0;
    float count = 0.0;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        if ( samples[i] < 0 )
            continue;

        mean += samples[i];
        count += 1.0;
    }
    mean /= count;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        if ( samples[i] < 0 )
            continue;

        float value = samples[i] - mean;
        stdv += (value*value);
    }
    stdv = sqrt(stdv/count);

    // take samples and update template descriptor
    int step = btd_filt->sample_step / 2;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        if ( stdv < 0.01 )
        {
            if ( samples[i] < 0 )
                continue;

            for ( int j = 0 ; j < num_bin ; j ++ )
            {
                btd->hist[i].num_sample += 1;
                btd->hist[i].value[j] += 1;
            }
            continue;
        }

        // sample from each region of a histogram
        for ( float dx = scale*(dir_x[i]-step) ; dx <= scale*(dir_x[i]+step) ; dx += 1 )
        {
            for ( float dy = scale*(dir_y[i]-step) ; dy <= scale*(dir_y[i]+step) ; dy += 1 )
            {
                // take a sample
                float sample = get_sample(image, stride, width, height, x+dx*c-dy*s, y+dx*s+dy*c);
                if ( sample < 0 ) continue;

                // normalize the pixel intensity value with the mean and standard deviation, to [0..255]
                double value = (((sample - mean)/stdv/2) + 1.0 ) * 128;
                if ( value < 0 ) value = 0;
                if ( value > 255 ) value = 255;

                // get the bin index from the value
                int bin = value * num_bin / 256;

                // update the histogram
                btd->hist[i].num_sample += 1;

                btd->hist[i].mean[bin] *= btd->hist[i].value[bin];
                btd->hist[i].mean[bin] += value;
                btd->hist[i].value[bin] += 1;
                btd->hist[i].mean[bin] /= btd->hist[i].value[bin];
            }
        }
    }

}

void BTD_learn_gradient( BTD_Filter * btd_filt, unsigned char * image, int stride, int width, int height, int x, int y, int scale, float angle, BTD * btd )
{
    int num_histogram = btd_filt->num_histogram;
    int num_bin = btd_filt->num_bin;

    // canonize orientation
    angle += get_orientation( image, stride, x, y );

    // get gradients from rotated sampling coordinates
    float c = cos(angle);
    float s = sin(angle);
    float * dir_x = btd_filt->hist_x;
    float * dir_y = btd_filt->hist_y;
    float grad_angle = 0.0f;
    float grad_mag = 0.0f;
    int step = btd_filt->sample_step / 2;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        // sample from each region of a histogram
        for ( float dx = scale*(dir_x[i]-step) ; dx <= scale*(dir_x[i]+step) ; dx += 1 )
        {
            for ( float dy = scale*(dir_y[i]-step) ; dy <= scale*(dir_y[i]+step) ; dy += 1 )
            {

                float ix00 = x + (  dx    * c -  dy    * s );
                float iy00 = y + (  dx    * s +  dy    * c );
                float ix10 = x + ( (dx+1) * c -  dy    * s );
                float iy10 = y + ( (dx+1) * s +  dy    * c );
                float ix01 = x + (  dx    * c - (dy+1) * s );
                float iy01 = y + (  dx    * s + (dy+1) * c );

                float sample_00 = get_sample( image, stride, width, height, ix00, iy00 );
                float sample_10 = get_sample( image, stride, width, height, ix10, iy10 );
                float sample_01 = get_sample( image, stride, width, height, ix01, iy01 );

                if ( sample_00 < 0 || sample_01 < 0 || sample_10 < 0 )
                {
                    grad_angle = 0;
                    grad_mag = 0;
                }
                else
                {
                    float dx_ = sample_10 - sample_00;
                    float dy_ = sample_01 - sample_00;
                    if ( dy_ < 0 ) { dx_ *= -1; dy_ *= -1; }
                    grad_mag = sqrt(dx_*dx_+dy_*dy_);
                    if ( grad_mag > 0 )
                    {
                        dx_ /= grad_mag;
                        grad_angle = atan2_table[(int)(dx_*atan2_table_size2+atan2_table_size2)];
                    }
                    //					grad_angle = atan2( dy_, dx_ )*180/M_PI;
                    //					while ( grad_angle < 0 ) grad_angle += 360;
                    //					grad_angle = ((int)grad_angle) % 180;
                    /*					while ( grad_angle >= 360 ) grad_angle -= 360;
                     if ( grad_angle > 180 ) grad_angle -= 180;*/
                    //					grad_mag = sqrt(dx_*dx_+dy_*dy_);
                }

                // NOTE: what should this threshold of magnitude be?
                if ( grad_mag < 3 )
                {
                    btd->hist[i].num_sample += 1;
                    btd->hist[i].value[0] += 1;
                    continue;
                }
                if ( grad_mag >= 30 ) grad_mag = 30;

                // get value from gradient orientation
                float value = grad_angle;
                // get the bin index from the value
                int bin = value * (num_bin-1) / 180 + 1;

                // update the histogram
                btd->hist[i].num_sample += grad_mag;

                btd->hist[i].mean[bin] *= btd->hist[i].value[bin];
                btd->hist[i].mean[bin] += value;
                btd->hist[i].value[bin] += grad_mag;
                btd->hist[i].mean[bin] /= btd->hist[i].value[bin];

            }
        }
    }

}

float BTD_test( BTD_Filter * btd_filt, BTD * btd_reference, unsigned char * image, int stride, int width, int height, int x, int y, float angle )
{
    float score = 0.0f;
    switch ( btd_filt->method )
    {
        case BTD_METHOD_INTENSITY:
            score = BTD_test_intensity( btd_filt, btd_reference, image, stride, width, height, x, y, angle );
            break;
        case BTD_METHOD_GRADIENT:
            score = BTD_test_gradient( btd_filt, btd_reference, image, stride, width, height, x, y, angle );
            break;
    }

    return score;
}

float BTD_test_intensity( BTD_Filter * btd_filt, BTD * btd_reference, unsigned char * image, int stride, int width, int height, int x, int y, float angle )
{
    float score = 0.0f;

    int num_histogram = btd_filt->num_histogram;
    int num_bin = btd_filt->num_bin;

    // calculate orientation
    float patch_angle = angle + get_orientation( image, stride, x, y );

    // get samples from rotated sampling coordinates
    float c = cos(patch_angle);
    float s = sin(patch_angle);
    float * dir_x = btd_filt->hist_x;
    float * dir_y = btd_filt->hist_y;
    int * samples = btd_filt->samples;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        if ( btd_reference->weight[i] > 0 )
        {
            float ix = x + ( dir_x[i] * c - dir_y[i] * s );
            float iy = y + ( dir_x[i] * s + dir_y[i] * c );

            samples[i] = get_sample( image, stride, width, height, ix, iy );
        }
        else {
            samples[i] = -1;
        }
    }

    // compute mean and standard deviation of the image region
    float mean = 0.0;
    float stdv = 0.0;
    float count = 0.0;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        if ( samples[i] < 0 )
            continue;

        mean += samples[i];
        count += 1.0;
    }
    mean /= count;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        if ( samples[i] < 0 )
            continue;

        float value = samples[i] - mean;
        stdv += (value*value);
    }
    stdv = sqrt(stdv/count);

    // checking too small standard deviation? this is uniform-colored region...
    if ( stdv < 1 ) return btd_reference->sum_weight;

    //	__m128i m1, m2, m3;

    // compare
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        if ( samples[i] < 0 )
            continue;

        // normalize the pixel intensity value with the mean and standard deviation, to [0..255]
        double value = (((samples[i] - mean)/stdv/2) + 1.0 ) * 128;
        if ( value < 0 ) value = 0;
        if ( value > 255 ) value = 255;

        // get the bin index from the value
        int bin = value * num_bin /256;

        // compare: SSE2
        //		m1 = _mm_set1_epi8( 1 << bin );
        //		m2 = _mm_set1_epi8( btd_reference->compact[i] );
        //		m3 = _mm_and_si128( m1, m2 );

        // compare with the reference
        //		if ( _mm_extract_epi16( m3, 0 ) )
        if ( (1<<bin) & btd_reference->compact[i] )
        {
            // NOTE: energy function?
            //			score += (num_bin/bit_count[btd_reference->compact[i]]);
            score += btd_reference->weight[i];
        }
    }

    return score;
}

float BTD_test_gradient( BTD_Filter * btd_filt, BTD * btd_reference, unsigned char * image, int stride, int width, int height, int x, int y, float angle )
{
    float score = 0.0f;

    int num_histogram = btd_filt->num_histogram;
    int num_bin = btd_filt->num_bin;

    // calculate orientation
    float patch_angle = angle + get_orientation( image, stride, x, y );

    // get gradients from rotated sampling coordinates
    float c = cos(patch_angle);
    float s = sin(patch_angle);
    float * dir_x = btd_filt->hist_x;
    float * dir_y = btd_filt->hist_y;
    float * grad_angle = btd_filt->angle;
    float * grad_mag = btd_filt->mag;
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        if ( btd_reference->weight[i] > 0 )
        {
            float ix00 = x + (  dir_x[i]    * c -  dir_y[i]    * s );
            float iy00 = y + (  dir_x[i]    * s +  dir_y[i]    * c );
            float ix10 = x + ( (dir_x[i]+1) * c -  dir_y[i]    * s );
            float iy10 = y + ( (dir_x[i]+1) * s +  dir_y[i]    * c );
            float ix01 = x + (  dir_x[i]    * c - (dir_y[i]+1) * s );
            float iy01 = y + (  dir_x[i]    * s + (dir_y[i]+1) * c );

            float sample_00 = get_sample( image, stride, width, height, ix00, iy00 );
            float sample_10 = get_sample( image, stride, width, height, ix10, iy10 );
            float sample_01 = get_sample( image, stride, width, height, ix01, iy01 );

            if ( sample_00 < 0 || sample_01 < 0 || sample_10 < 0 )
            {
                grad_angle[i] = 0;
                grad_mag[i] = -1;
            }
            else
            {
                float dx = sample_10 - sample_00;
                float dy = sample_01 - sample_00;
                if ( dy < 0 ) { dx *= -1; dy *= -1; }
                grad_mag[i] = sqrt(dx*dx+dy*dy);
                if ( grad_mag[i] > 0 )
                {
                    dx /= grad_mag[i];
                    grad_angle[i] = atan2_table[(int)(dx*atan2_table_size2+atan2_table_size2)];
                }
                //				grad_angle[i] = atan2( dy, dx )*180/M_PI;
                //				while ( grad_angle[i] < 0 ) grad_angle[i] += 360;
                //				grad_angle[i] = (int)(grad_angle[i]) % 180;
                /*				while ( grad_angle[i] >= 360 ) grad_angle[i] -= 360;
                 if ( grad_angle[i] > 180 ) grad_angle[i] -= 180; */
                //				grad_mag[i] = sqrt(dx*dx+dy*dy);
            }
        }
        else {
            grad_angle[i] = 0;
            grad_mag[i] = -1;
        }

    }


    //	__m128i m1, m2, m3;

    // compare
    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        if ( grad_mag[i] < 0 )
            continue;

        // get the bin index
        int bin = 0;
        if ( grad_mag[i] >= 3 )
        {
            bin = grad_angle[i] * (num_bin-1) / 180 + 1;
        }

        // compare: SSE2
        //		m1 = _mm_set1_epi8( 1 << bin );
        //		m2 = _mm_set1_epi8( btd_reference->compact[i] );
        //		m3 = _mm_and_si128( m1, m2 );

        // compare with the reference
        //		if ( _mm_extract_epi16( m3, 0 ) )
        if ( (1<<bin) & btd_reference->compact[i] )
        {
            // NOTE: energy function?
            //			score += (num_bin/bit_count[btd_reference->compact[i]]);
            score += btd_reference->weight[i];
        }
    }

    return score;
}

void BTD_make_compact_reference( BTD_Filter * btd_filt, BTD * btd )
{
    memset( btd->compact, 0, sizeof(unsigned char) * btd_filt->num_histogram );

    for ( int i = 0 ; i < btd_filt->num_histogram ; i ++ )
    {
        float threshold = 0.05 * btd->hist[i].num_sample;
        if ( threshold <= 0 )
        {
            btd->compact[i] = 0xFF;
        }
        else
        {
            for ( int j = 0 ; j < btd_filt->num_bin ; j ++ )
            {
                if ( btd->hist[i].value[j] <= threshold )
                {
                    btd->compact[i] |= ( 1 << j );
                }
            }
        }
    }
}

void BTD_make_compact_sample( BTD_Filter * btd_filt, BTD * btd )
{
    memset( btd->compact, 0, sizeof(unsigned char) * btd_filt->num_histogram );

    for ( int i = 0 ; i < btd_filt->num_histogram ; i ++ )
    {
        //		float threshold = 0.05 * btd->hist[i].num_sample;
        for ( int j = 0 ; j < btd_filt->num_bin ; j ++ )
        {
            if ( btd->hist[i].value[j] > 0 )
            {
                btd->compact[i] |= ( 1 << j );
            }
        }
    }
}


int BTD_compare_compact( BTD_Filter * btd_filt, BTD * btd_sample, BTD * btd_reference )
{
    int num_histogram = btd_filt->num_histogram;
    unsigned char * sample = btd_sample->compact;
    unsigned char * reference = btd_reference->compact;
    float * weight = btd_reference->weight;
    float score = 0;
//    int value;

    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        if ( sample[i] & reference[i] )
        {
            score += weight[i];
        }
        /*		if ( weight[i] > 0 )
         {
         value = sample[i] & reference[i];
         score += ( bit_count[ value ] * weight[i] );
         }*/
    }

    return (int)score;
}

float BTD_compare( BTD_Filter * btd_filt, BTD * btd_sample, BTD * btd_reference )
{
    int num_histogram = btd_filt->num_histogram;
    int num_bin = btd_filt->num_bin;
    float score = 0.0f;

    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        Histogram * hist_ref = &(btd_reference->hist[i]);
        Histogram * hist_sample = &(btd_sample->hist[i]);

        if ( hist_ref->num_sample > 0 && hist_sample->num_sample > 0 )
        {
            for ( int j = 0 ; j < num_bin ; j ++ )
            {
                // compare two histograms !!!
                score += (hist_sample->value[j] * hist_ref->value[j] / (float)hist_ref->num_sample);
            }
        }
    }
    score = num_histogram - score;

    return score;
}

float BTD_quality( BTD_Filter * btd_filt, BTD * btd )
{
    float quality = 0.0f;
    int num_histogram = btd_filt->num_histogram;
    int num_bin = btd_filt->num_bin;

    for ( int i = 0 ; i < num_histogram ; i ++ )
    {
        int threshold = 0.05 * btd->hist[i].num_sample;
        int num_empty_bin = 0;
        for ( int j = 0 ; j < num_bin ; j ++ )
        {
            if ( btd->hist[i].value[j] <= threshold )
            {
                num_empty_bin ++;
            }
        }
        if ( num_empty_bin > 0.5 * num_bin )
        {
            quality += 1.0;
        }
    }
    quality /= (float)num_histogram;

    return quality;
}

float get_orientation( unsigned char * image, int stride, int x, int y )
{
    return 0;
    float dx = 0.0 ,dy = 0.0;
    const int dir_x[16] = {0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
    const int dir_y[16] = {3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1,0,1,2,3};

    for ( int i = 0 ; i < 8 ; i ++ )
    {
        int value0 = image[(y+dir_y[i  ])*stride+(x+dir_x[i  ])];
        int value1 = image[(y+dir_y[i+8])*stride+(x+dir_x[i+8])];
        dx += (dir_x[i]-dir_x[i+8])*(value0-value1);
        dy += (dir_y[i]-dir_y[i+8])*(value0-value1);
    }

    return atan2( dy, dx );
}

float get_sample( unsigned char * image, int stride, int width, int height, float x, float y )
{
    /*
     // this is a simple sampling
     int ix = (int)(x+0.5);
     int iy = (int)(y+0.5);
     int value = -1;
     if ( ix >= 0 && ix < width && iy >= 0 && iy < height )
     {
     value = image[iy*stride+ix];
     }
     */


    // this is an interpolation sampling
    int ix0 = (int)x;
    int iy0 = (int)y;
    int ix1 = ix0+1;
    int iy1 = iy0+1;

    if ( ix0 < 0 || ix1 >= width || iy0 < 0 || iy1 >= height )
        return -1;

    float ratio_x1 = x - ix0;
    float ratio_y1 = y - iy0;

    float value0 = image[iy0*stride+ix0] * (1-ratio_x1) + image[iy0*stride+ix1] * ratio_x1;
    float value1 = image[iy1*stride+ix0] * (1-ratio_x1) + image[iy1*stride+ix1] * ratio_x1;
    float value = value0 * (1-ratio_y1) + value1 * ratio_y1;
    /**/
    return value;
}

MTD * MTD_create( int num_template, int num_anchor, BTD_Filter ** btd_filt )
{
    MTD * mtd = (MTD *)malloc( sizeof(MTD) );
    mtd->num_template = num_template;
    mtd->num_anchor = num_anchor;

    mtd->btd_filt = (BTD_Filter **)malloc( sizeof(BTD_Filter *) * num_template );
    mtd->btd = (BTD **)malloc( sizeof(BTD *) * num_template );
    mtd->weight = (float *)malloc( sizeof(float) * num_template );
    for ( int i = 0 ; i < num_template ; i ++ )
    {
        mtd->btd_filt[i] = btd_filt[i];
        mtd->btd[i] = BTD_create( btd_filt[i] );
        mtd->weight[i] = 1.0/(float)num_template;
    }

    mtd->anchor = (int *)malloc( sizeof(int) * num_anchor );
    mtd->feat_dist = (int *)malloc( sizeof(int) * btd_filt[0]->num_histogram );
    memset( mtd->feat_dist, 0, sizeof(int) * btd_filt[0]->num_histogram );

    return mtd;
}

void MTD_free( MTD ** mtd )
{
    for ( int i = 0 ; i < (*mtd)->num_template ; i ++ )
    {
        if ( (*mtd)->btd[i] )
        {
            BTD_free( (*mtd)->btd_filt[i], &((*mtd)->btd[i]) );
        }
    }
    free( (*mtd)->btd );
    free( (*mtd)->btd_filt );
    free( (*mtd)->weight );
    free( (*mtd)->anchor );
    free( (*mtd)->feat_dist );

    free( (*mtd) );
    *mtd = 0;
}

void MTD_learn( MTD * mtd, unsigned char * image, int stride, int width, int height, int x, int y, int scale, float angle )
{
    for ( int i = 0 ; i < mtd->num_template ; i ++ )
    {
        // update template descriptors
        BTD_learn( mtd->btd_filt[i], image, stride, width, height, x, y, scale, angle, mtd->btd[i] );

        // make compact representations
        BTD_make_compact_reference( mtd->btd_filt[i], mtd->btd[i] );

        // compute histogram weights
        mtd->btd[i]->sum_weight = 0.0f;
        if ( mtd->btd_filt[i]->method == BTD_METHOD_INTENSITY )
        {
            for ( int j = 0 ; j < mtd->btd_filt[i]->num_histogram ; j ++ )
            {
                mtd->btd[i]->weight[j] = (bit_count[mtd->btd[i]->compact[j]] > 0.3 * mtd->btd_filt[i]->num_bin );
                mtd->btd[i]->sum_weight += mtd->btd[i]->weight[j];
            }
        }
        else if ( mtd->btd_filt[i]->method == BTD_METHOD_GRADIENT )
        {
            for ( int j = 0 ; j < mtd->btd_filt[i]->num_histogram ; j ++ )
            {
                mtd->btd[i]->weight[j] = ( mtd->btd[i]->hist[j].value[0]/mtd->btd[i]->hist[j].num_sample < 0.5 );/*&&
                                                                                                                  bit_count[mtd->btd[i]->compact[j]] > 0.2 * (mtd->btd_filt[i]->num_bin-1) );*/
                mtd->btd[i]->sum_weight += mtd->btd[i]->weight[j];
            }
        }
    }
}

void MTD_update( MTD * mtd, unsigned char * image, int stride, int width, int height, int x, int y, float angle0 )
{
    for ( int i = 0 ; i < mtd->num_template ; i ++ )
    {
        // update template descriptors
        BTD_update( mtd->btd_filt[i], image, stride, width, height, x, y, mtd->btd[i], angle0 );

        // make compact representations
        BTD_make_compact_reference( mtd->btd_filt[i], mtd->btd[i] );

        // compute histogram weights
        mtd->btd[i]->sum_weight = 0.0f;
        if ( mtd->btd_filt[i]->method == BTD_METHOD_INTENSITY )
        {
            for ( int j = 0 ; j < mtd->btd_filt[i]->num_histogram ; j ++ )
            {
                mtd->btd[i]->weight[j] = (bit_count[mtd->btd[i]->compact[j]] > 0.5 * mtd->btd_filt[i]->num_bin);
                mtd->btd[i]->sum_weight += mtd->btd[i]->weight[j];
            }
        }
        else if ( mtd->btd_filt[i]->method == BTD_METHOD_GRADIENT )
        {
            for ( int j = 0 ; j < mtd->btd_filt[i]->num_histogram ; j ++ )
            {
                mtd->btd[i]->weight[j] = (mtd->btd[i]->hist[j].value[0]/mtd->btd[i]->hist[j].num_sample < 0.2 );/* &&
                                                                                                                 bit_count[mtd->btd[i]->compact[j]] > 0.2 * mtd->btd_filt[i]->num_bin);*/
                mtd->btd[i]->sum_weight += mtd->btd[i]->weight[j];
            }
        }
    }
}

void MTD_sample( MTD * mtd, unsigned char * image, int stride, int width, int height, int x, int y, float angle0 )
{
    for ( int i = 0 ; i < mtd->num_template ; i ++ )
    {
        BTD_clear( mtd->btd_filt[i], mtd->btd[i] );
        BTD_update( mtd->btd_filt[i], image, stride, width, height, x, y, mtd->btd[i], angle0 );
        BTD_make_compact_sample( mtd->btd_filt[i], mtd->btd[i] );
    }
}

float MTD_test( MTD * mtd, unsigned char * image, int stride, int width, int height, int x, int y, float angle, float threshold )
{
    float score = 0.0f;

    for ( int i = 0 ; score < threshold && i < mtd->num_template ; i ++ )
    {
        score += ( mtd->weight[i] * BTD_test( mtd->btd_filt[i], mtd->btd[i], image, stride, width, height, x, y, angle ) / mtd->btd[i]->sum_weight );
    }

    return score;
}

void MTD_update_feat_dist( MTD * mtd, int x, int y )
{
    float row = y / mtd->btd_filt[0]->sample_step + mtd->btd_filt[0]->num_row/2;
    float col = x / mtd->btd_filt[0]->sample_step + mtd->btd_filt[0]->num_col/2;

    int index = col + row * mtd->btd_filt[0]->num_col;
    mtd->feat_dist[index] ++;
}

void MTD_update_feat_dist( MTD * mtd, int cx, int cy, int x, int y, int scale )
{
    int row = (x - cx)/mtd->btd_filt[0]->sample_step/scale + mtd->btd_filt[0]->num_row/2;
    int col = (y - cy)/mtd->btd_filt[0]->sample_step/scale + mtd->btd_filt[0]->num_col/2;

    int index = col + row * mtd->btd_filt[0]->num_col;
    mtd->feat_dist[index] ++;
}

void MTD_select_anchors( MTD * mtd )
{
    int best_max = 100000;
    int best = 1;
    int best_index = -1;

    // default, the center point as an anchor
    //	mtd->anchor[0] = mtd->btd_filt[0]->num_histogram/2;

    // add more anchors
    for ( int i = 0 ; i < mtd->num_anchor ; i ++ )
    {
        best = 1;
        best_index = -1;
        for ( int j = 0 ; j < mtd->btd_filt[0]->num_histogram ; j ++ )
        {
            if ( mtd->feat_dist[j] > best && mtd->feat_dist[j] <= best_max )
            {
                bool good = true;
                for ( int k = 0 ; k < i ; k ++ )
                {
                    if ( mtd->anchor[k] == j )
                        good = false;
                }
                if ( good )
                {
                    best = mtd->feat_dist[j];
                    best_index = j;
                }
            }
        }
        if ( best_index >= 0 )
        {
            mtd->anchor[i] = best_index;
            best_max = mtd->feat_dist[best_index];
        }
    }
}

void MTD_make_compact_reference( MTD * mtd )
{
    for ( int i = 0 ; i < mtd->num_template ; i ++ )
    {
        BTD_make_compact_reference( mtd->btd_filt[i], mtd->btd[i] );
    }
}

void MTD_make_compact_sample( MTD * mtd )
{
    for ( int i = 0 ; i < mtd->num_template ; i ++ )
    {
        BTD_make_compact_sample( mtd->btd_filt[i], mtd->btd[i] );
    }
}

float MTD_compare( MTD * mtd_sample, MTD * mtd_reference )
{
    float score = 0.0f;

    for ( int i = 0 ; i < mtd_reference->num_template ; i ++ )
    {
        score += ( mtd_reference->weight[i] * BTD_compare( mtd_sample->btd_filt[i], mtd_sample->btd[i], mtd_reference->btd[i] ) / mtd_reference->btd[i]->sum_weight );
    }

    return score;
}

float MTD_compare_compact( MTD * mtd_sample, MTD * mtd_reference )
{
    float score = 0.0f;

    for ( int i = 0 ; i < mtd_reference->num_template ; i ++ )
    {
        score += ( mtd_reference->weight[i] * BTD_compare_compact( mtd_sample->btd_filt[i], mtd_sample->btd[i], mtd_reference->btd[i] ) / mtd_reference->btd[i]->sum_weight );
    }

    return score;
}
