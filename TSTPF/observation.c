//
//  observation.c
//  TSTPF
//
//  Created by BeiXinzhu on 4/7/16.
//  Copyright © 2016 BeiXinzhu. All rights reserved.
//

#include "defs.h"
//#include "utils.h"
#include "observation.h"
#include "opencv/cv.h"
#include "particles.h"



/*
 Converts a BGR image to HSV colorspace

 @param bgr image to be converted

 @return Returns bgr converted to a 3-channel, 32-bit HSV image with
 S and V values in the range [0,1] and H value in the range [0,360]
 */
IplImage* bgr2hsv( IplImage* bgr )
{
    IplImage* bgr32f, * hsv;
    bgr32f = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_32F, 3 );
    hsv = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_32F, 3 );
    cvConvertScale( bgr, bgr32f, 1.0 / 255.0, 0 );
    cvCvtColor( bgr32f, hsv, CV_BGR2HSV );
    cvReleaseImage( &bgr32f );
    return hsv;
}


/*
 Calculates the histogram bin into which an HSV entry falls

 @param h Hue
 @param s Saturation
 @param v Value

 @return Returns the bin index corresponding to the HSV color defined by
 \a h, \a s, and \a v.
 */
int histo_bin( float h, float s, float v )
{
    int hd, sd, vd;

    /* if S or V is less than its threshold, return a "colorless" bin */
    vd = MIN( (int)(v * NV / V_MAX), NV-1 );
    if( s < S_THRESH  ||  v < V_THRESH )
        return NH * NS + vd;

    /* otherwise determine "colorful" bin */
    hd = MIN( (int)(h * NH / H_MAX), NH-1 );
    sd = MIN( (int)(s * NS / S_MAX), NS-1 );
    return sd * NH + hd;
}



/*
 Calculates a cumulative histogram as defined above for a given array
 of images

 @param img an array of images over which to compute a cumulative histogram;
 each must have been converted to HSV colorspace using bgr2hsv()
 @param n the number of images in imgs

 @return Returns an un-normalized HSV histogram for \a imgs
 */
histogram* calc_histogram( IplImage** imgs, int n )
{
    IplImage* img;
    histogram* histo;
    IplImage* h, * s, * v;
    float* hist;
    int i, r, c, bin;

    histo = malloc( sizeof(histogram) );
    histo->n = NH*NS + NV;
    hist = histo->histo;
    memset( hist, 0, histo->n * sizeof(float) );

    for( i = 0; i < n; i++ )
    {
        /* extract individual HSV planes from image */
        img = imgs[i];
        h = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
        s = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
        v = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
        //cvCvtPixToPlane( img, h, s, v, NULL );
        cvSplit( img, h, s, v, NULL );


        /* increment appropriate histogram bin for each pixel */
        for( r = 0; r < img->height; r++ )
            for( c = 0; c < img->width; c++ )
            {
                bin = histo_bin( /*pixval32f( h, r, c )*/((float*)(h->imageData + h->widthStep*r) )[c],
                                ((float*)(s->imageData + s->widthStep*r) )[c],
                                ((float*)(v->imageData + v->widthStep*r) )[c] );
                hist[bin] += 1;
            }
        cvReleaseImage( &h );
        cvReleaseImage( &s );
        cvReleaseImage( &v );
    }
    return histo;
}



/*
 Normalizes a histogram so all bins sum to 1.0

 @param histo a histogram
 */
void normalize_histogram( histogram* histo )
{
    float* hist;
    float sum = 0, inv_sum;
    int i, n;

    hist = histo->histo;
    n = histo->n;

    /* compute sum of all bins and multiply each bin by the sum's inverse */
    for( i = 0; i < n; i++ )
        sum += hist[i];
    inv_sum = 1.0 / sum;
    for( i = 0; i < n; i++ )
        hist[i] *= inv_sum;
}



/*
 Computes squared distance metric based on the Battacharyya similarity
 coefficient between histograms.

 @param h1 first histogram; should be normalized
 @param h2 second histogram; should be normalized

 @return Returns a squared distance based on the Battacharyya similarity
 coefficient between \a h1 and \a h2
 */
float histo_dist_sq( histogram* h1, histogram* h2 )
{
    float* hist1, * hist2;
    float sum = 0;
    int i, n;

    n = h1->n;
    hist1 = h1->histo;
    hist2 = h2->histo;

    /*
     According the the Battacharyya similarity coefficient,

     D = \sqrt{ 1 - \sum_1^n{ \sqrt{ h_1(i) * h_2(i) } } }
     */
    for( i = 0; i < n; i++ )
        sum += sqrt( hist1[i]*hist2[i] );
    return 1.0 - sum;
}



/*
 Computes the likelihood of there being a player at a given location in
 an image

 @param img image that has been converted to HSV colorspace using bgr2hsv()
 @param r row location of center of window around which to compute likelihood
 @param c col location of center of window around which to compute likelihood
 @param w width of region over which to compute likelihood
 @param h height of region over which to compute likelihood
 @param ref_histo reference histogram for a player; must have been
 normalized with normalize_histogram()

 @return Returns the likelihood of there being a player at location
 (\a r, \a c) in \a img
 */
float likelihood( IplImage* img, int r, int c,
                 int w, int h, histogram* ref_histo )
{
    IplImage* tmp;
    histogram* histo;
    float d_sq;

    /* extract region around (r,c) and compute and normalize its histogram */
    
    CvRect rect = cvRect( c - w / 2, r - h / 2, w, h );
    if (rect.width >= 0 && rect.height >= 0 && rect.x < img->width && rect.y < img->height && rect.x + rect.width >= (int)(rect.width > 0) && rect.y + rect.height >= (int)(rect.height > 0)){
        cvSetImageROI( img, cvRect( c - w / 2, r - h / 2, w, h ) );
        tmp = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 3 );
        cvCopy( img, tmp, NULL );
        cvResetImageROI( img );
        histo = calc_histogram( &tmp, 1 );
        cvReleaseImage( &tmp );
        normalize_histogram( histo );

        /* compute likelihood as e^{\lambda D^2(h, h^*)} */
        d_sq = histo_dist_sq( histo, ref_histo );
        free( histo );
        return exp( -LAMBDA * d_sq );
    }
    else {
        printf("err!\n");
        return -1;
    }
}



/*
 Returns an image containing the likelihood of there being a player at
 each pixel location in an image

 @param img the image for which likelihood is to be computed; should have
 been converted to HSV colorspace using bgr2hsv()
 @param w width of region over which to compute likelihood
 @param h height of region over which to compute likelihood
 @param ref_histo reference histogram for a player; must have been
 normalized with normalize_histogram()

 @return Returns a single-channel, 32-bit floating point image containing
 the likelihood of every pixel location in \a img normalized so that the
 sum of likelihoods is 1.
 */
IplImage* likelihood_image( IplImage* img, int w, int h, histogram* ref_histo )
{
    IplImage* l;
    CvScalar sum;
    int i, j;

    l = cvCreateImage( cvGetSize( img ), IPL_DEPTH_32F, 1 );
    for( i = 0; i < img->height; i++ )
        for( j = 0; j < img->width; j++ )
            //setpix32f( l, i, j, likelihood( img, i, j, w, h, ref_histo ) );
            ( (float*)(l->imageData + l->widthStep*i) )[j] = likelihood( img, i, j, w, h, ref_histo ) ;

    sum = cvSum( l );
    cvScale( l, l, 1.0 / sum.val[0], 0 );
    return l;
}



/*
 Exports histogram data to a specified file.  The file is formatted as
 follows (intended for use with gnuplot:

 0 <h_0>
 ...
 <i> <h_i>
 ...
 <n> <h_n>

 Where n is the number of histogram bins and h_i, i = 1..n are
 floating point bin values

 @param histo histogram to be exported
 @param filename name of file to which histogram is to be exported

 @return Returns 1 on success or 0 on failure
 */
int export_histogram( histogram* histo, char* filename )
{
    int i, n;
    float* h;
    FILE* file = fopen( filename, "w" );

    if( ! file )
        return 0;
    n = histo->n;
    h = histo->histo;
    for( i = 0; i < n; i++ )
        fprintf( file, "%d %f\n", i, h[i] );
    fclose( file );
    return 1;
}

// int getregions(IplImage* frame, CvRect** regions){
//     CvRect* r;
//     int x1, y1, w, h;
//
//     r = malloc( 1 * sizeof( CvRect ) );
//
//     int width = frame->width;
//     int height = frame->height;
//     int xcenter = width/2;
//     int ycenter = height/2;
//
//     x1 = round(xcenter - xcenter/2);
//     y1 = round(ycenter - ycenter/2);
//
//     w = xcenter /2;
//     h = ycenter / 2;
//
//     w = ( w % 2 )? w : w+1;
//     h = ( h % 2 )? h : h+1;
//     r[0] = cvRect( x1, y1, w, h );
//
//     *regions = r;
//     return 1;
// }

histogram** compute_ref_histos( IplImage* frame, CvRect* regions, int n )
{
    histogram** histos = malloc( n * sizeof( histogram* ) );
    IplImage* tmp;
    int i;

    /* extract each region from frame and compute its histogram */
    for( i = 0; i < n; i++ )
    {
        cvSetImageROI( frame, regions[i] );
        tmp = cvCreateImage( cvGetSize( frame ), IPL_DEPTH_32F, 3 );
        cvCopy( frame, tmp, NULL );
        cvResetImageROI( frame );
        histos[i] = calc_histogram( &tmp, 1 );
        normalize_histogram( histos[i] );
        cvReleaseImage( &tmp );
    }

    return histos;
}

IplImage * change4channelTo3InIplImage(IplImage * src) {
    if (src->nChannels != 4) {
        return NULL;
    }
    int row;
    int col;

    IplImage * destImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3);
    for (row = 0; row < src->height; row++) {
        for (col = 0; col < src->width; col++) {
            CvScalar s = cvGet2D(src, row, col);
            cvSet2D(destImg, row, col, s);
        }
    }

    return destImg;
}