//
//  particles.c
//  TSTPF
//
//  Created by BeiXinzhu on 4/7/16.
//  Copyright © 2016 BeiXinzhu. All rights reserved.
//

#include "defs.h"
//#include "utils.h"
#include "particles.h"
#include "time.h"
#include <stdio.h>
#include <math.h>
#include <opencv/highgui.h>

#define BB0  1

#define TRANS_X_STD /*5.0*/ 20
#define TRANS_Y_STD /*2.5*/ 20
#define X_init_STD /*5.0*/ 100
#define Y_init_STD /*2.5*/ 100
#define TRANS_S_STD 0.01

#define EPSILON 2

/* autoregressive dynamics parameters for transition model */
#define A1 /* 2.0*/ 2
#define A2 /*-1.0*/ -1
#define B0  1.0000

particle* init_distribution( CvRect* regions, histogram** histos, int n, int p, int w, int h, float U0, float * track_score, int num_match, estimate estm)
{
    particle* particles;
    float x[num_match], y[num_match];
    int i, j, k = 0;
    float temp1,temp2;
    int width[num_match];
    int height[num_match];
    float np[num_match];
    int selectregion;

    particles = malloc( p * sizeof( particle ) );
    // np = p / n;
    float sum_score;
    for (i = 0; i < num_match; i++){
      sum_score += track_score[i];
    }

    np[0] = track_score[0] / sum_score;
    for (i = 1; i < num_match; i++){
      np[i] = np[i-1] + track_score[i] / sum_score;
    }

    for (i = 0; i < num_match; i++){
      width[i] = regions[i].width;
      height[i] = regions[i].height;
      x[i] = regions[i].x + width[i] / 2;
      y[i] = regions[i].y + height[i] / 2;
    }

    for (i = 0;i < p; i++){
      temp1 = rand() / (float)(RAND_MAX);
      selectregion = -1;
      for (j = 0; j < num_match; j++){
        if (temp1 <= np[j]) {
          selectregion = j;
          break;
        }
      }

      if (selectregion == -1) selectregion = 0;

      temp2 = rand() / (float)(RAND_MAX);

      if (temp2<=U0){
        //   particles[k].x0 = x[selectregion] + B0 * gaussrand(0,X_init_STD);
        //   particles[k].y0 = y[selectregion] + B0 * gaussrand(0,Y_init_STD);
          //
        //   particles[k].x0 = particles[k].xp = particles[k].x = MAX( 0.0, MIN( (float)w - 1.0, particles[k].x0 ) );
        //   particles[k].y0 = particles[k].yp = particles[k].y = MAX( 0.0, MIN( (float)h - 1.0, particles[k].y0 ) );
            particles[k].x = MAX( 0.0, MIN( (float)w - 1.0, x[selectregion] ) );
            particles[k].y = MAX( 0.0, MIN( (float)h - 1.0, y[selectregion] ) );
            particles[k].v_x = estm.velocity_x;
            particles[k].v_y = estm.velocity_y;
            particles[k].av_x = estm.accelerate_vx;
            particles[k].av_y = estm.accelerate_vy;
            // particles[k].xp = particles[k].x - estm.velocity_x;
            // particles[k].yp = particles[k].y - estm.velocity_y;
            // particles[k].xp = MAX( 0.0, MIN( (float)w - 1.0, particles[k].xp ) );
            // particles[k].yp = MAX( 0.0, MIN( (float)h - 1.0, particles[k].yp ) );
          particles[k].sp = particles[k].s = 1.0;
          particles[k].width = width[selectregion];
          particles[k].height = height[selectregion];
        //   particles[k].width = estm->width;
        //   particles[k].height = estm->height;
          particles[k].histo = histos[0];
          particles[k].alive = 1;
          particles[k++].w = 0;
      }
      else{
        //   particles[k].xp = particles[k].x = 0.0;
        //   particles[k].yp = particles[k].y = 0.0;
          particles[k].x = particles[k].y = 0.0;
          particles[k].v_x = particles[k].v_y = 0.0;
          particles[k].av_x = particles[k].av_y = 0.0;
          particles[k].sp = particles[k].s = 0.0;
          particles[k].width = width[selectregion];
          particles[k].height = height[selectregion];
          particles[k].histo = histos[0];
          particles[k].alive = 0;
          particles[k++].w = 0;
      }

    }

    /* make sure to create exactly p particles */
//    i = 0;
    while( k < p )
    {
        selectregion = rand() % num_match;
        particles[k].x = MAX( 0.0, MIN( (float)w - 1.0, x[selectregion] ) );
        particles[k].y = MAX( 0.0, MIN( (float)h - 1.0, y[selectregion] ) );
        particles[k].v_x = estm.velocity_x;
        particles[k].v_y = estm.velocity_y;
        particles[k].av_x = estm.accelerate_vx;
        particles[k].av_y = estm.accelerate_vy;
        // particles[k].xp = particles[k].x - estm.velocity_x;
        // particles[k].yp = particles[k].y - estm.velocity_y;
        // particles[k].xp = MAX( 0.0, MIN( (float)w - 1.0, particles[k].xp ) );
        // particles[k].yp = MAX( 0.0, MIN( (float)h - 1.0, particles[k].yp ) );
        particles[k].sp = particles[k].s = 1.0;
        particles[k].width = width[selectregion];
        particles[k].height = height[selectregion];
        //   particles[k].width = estm->width;
        //   particles[k].height = estm->height;
        particles[k].histo = histos[0];
        particles[k++].w = 0;
        particles[k].alive = 1;
    }

    return particles;
}

double gaussrand(double E, double V)
{
    static double V1, V2, S;
    static int phase = 0;
    double X;
    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
    phase = 1 - phase;
    X = X * V + E;
    return X;
}


/*
 Samples a transition model for a given particle

 @param p a particle to be transitioned
 @param w video frame width
 @param h video frame height
 @param rng a random number generator from which to sample

 @return Returns a new particle sampled based on <EM>p</EM>'s transition
 model
 */
particle transition( particle p, int w, int h,float U0,float U1, CvRect* regions,histogram** histos ,float * np, int * xx, int * yy, int * ww, int * hh , int num_match, estimate estm)
{
    //    srand(time(0));
    float x, y, s;
    float temp1;
    particle pn;
    int selectregion = -1;

    if (p.alive ==0 && (rand()/(float)(RAND_MAX))<=U0){
        pn.alive = 1;

        temp1= rand() / (float) (RAND_MAX);
        for (int i = 0; i < num_match; i++){
          if(temp1 <= np[i]){
            selectregion = i;
            break;
          }
        }

        if (selectregion == -1) selectregion = 0;

        pn.x = MAX( 0.0, MIN( (float)w - 1.0, xx[selectregion]+ B0 * gaussrand(0,X_init_STD) ) );
        pn.y = MAX( 0.0, MIN( (float)h - 1.0, yy[selectregion]+ B0 * gaussrand(0,Y_init_STD) ) );
        pn.v_x = estm.velocity_x;
        pn.v_y = estm.velocity_y;
        pn.av_x = estm.accelerate_vx;
        pn.av_y = estm.accelerate_vy;
        // pn.xp = pn.x - estm.velocity_x;
        // pn.yp = pn.y - estm.velocity_y;
        // pn.xp = MAX( 0.0, MIN( (float)w - 1.0, pn.xp ) );
        // pn.yp = MAX( 0.0, MIN( (float)h - 1.0, pn.yp ) );
        pn.sp = pn.s = 1.0;
        pn.width = ww[selectregion];
        pn.height = hh[selectregion];
        pn.histo = histos[0];
        pn.alive = 1;
        pn.w = 0;

    }//born
    else if (p.alive == 1 && (rand()/(float)(RAND_MAX))<=1-U1){
        x = p.x + p.v_x + p.av_x + B0 * gaussrand(0, TRANS_X_STD);
        pn.x = MAX( 0.0, MIN( (float)w - 1.0, x ) );
        y = p.y + p.v_y + p.av_y + B0 * gaussrand(0, TRANS_Y_STD);
        pn.y = MAX( 0.0, MIN( (float)h - 1.0, y ) );
        s = p.s + 1 * (p.s - p.sp) + B0 * gaussrand(0, TRANS_S_STD);
//        s = A1 * ( p.s - 1.0 ) + A2 * ( p.sp - 1.0 ) +
//        B0 * gaussrand(0, TRANS_S_STD) + 1.0;
        pn.s = MAX( 0.1, s );
        pn.v_x = pn.x - p.x;
        pn.v_y = pn.y - p.y;
        pn.av_x = pn.v_x - p.v_x;
        pn.av_y = pn.v_y - p.v_y;
        // pn.xp = p.x;
        // pn.yp = p.y;
        pn.sp = p.s;
        pn.width = p.width;
        pn.height = p.height;
        pn.histo = p.histo;
        pn.w = 0;
        pn.alive = 1;

    }//maintain
    else {
        // pn.xp = pn.x = 0.0;
        // pn.yp = pn.y = 0.0;
        pn.x = pn.y = 0.0;
        pn.v_x = pn.v_y = 0.0;
        pn.av_x = pn.av_y = 0.0;
        pn.sp = pn.s = 0.0;
        pn.width = 0.0;
        pn.height = 0.0;
        pn.histo = histos[0];
        pn.alive = 0;
        pn.w = 0;
    }//dead

    /* sample new state using second-order autoregressive dynamics */

    return pn;
}



/*
 Normalizes particle weights so they sum to 1

 @param particles an array of particles whose weights are to be normalized
 @param n the number of particles in \a particles
 */
void normalize_weights( particle* particles, int n )
{
    float sum = 0;
    int i;

    for( i = 0; i < n; i++ )
        if (particles[i].alive == 1)
            sum += particles[i].w;
    for( i = 0; i < n; i++ ){
        if (particles[i].alive == 1)
            particles[i].w /= sum;
        else
            particles[i].w =0;
    }

}



/*
 Re-samples a set of particles according to their weights to produce a
 new set of unweighted particles

 @param particles an old set of weighted particles whose weights have been
 normalized with normalize_weights()
 @param n the number of particles in \a particles

 @return Returns a new set of unweighted particles sampled from \a particles
 */
particle* resample( particle* particles, int n )
{
    particle* new_particles;
    int i, j, np, k = 0;

    qsort( particles, n, sizeof( particle ), &particle_cmp );
    new_particles = malloc( n * sizeof( particle ) );
    for( i = 0; i < n; i++ )
        //    for( i = n-1; i >0; i-- )
    {
        np = round( particles[i].w * n );
        for( j = 0; j < np; j++ )
        {
            new_particles[k++] = particles[i];
            new_particles[k].w = 0;
            if( k == n )
                goto exit;
        }
    }
    while( k < n )
        new_particles[k++] = particles[0];

exit:
    return new_particles;
}

particle* resample1( particle* particles, int n )
{
    //    srand(time(0));
    particle* new_particles;
    int i, j, k = 0;
    float u;
    new_particles = malloc( n * sizeof( particle ) );

    float weightsum[n+1];
    weightsum[0] = 0;
    for( i = 1; i <=n ; i++ ){
        weightsum[i] = weightsum[i-1] + particles[i].w;
    }
    u = 0;k=0;
    while (k<n) {
        u = rand() / (double)(RAND_MAX);
        while (u==0) {
            u = rand() / (double)(RAND_MAX);
        }
        //        printf("%d = %f\n",k,u);
        for(j = 1;j<=n;j++){
            if (u>weightsum[j-1] && u<=weightsum[j]) {
                new_particles[k++] = particles[j-1];
                new_particles[k].w = 0;
                //                printf("j = %d\n", j);
                break;
            }
        }

    }
    return new_particles;
}

particle* resample2( particle* particles, int n ,int num_particles)
{
    //    srand(time(0));
    particle* new_particles;
    int i, j = 0;
    new_particles = malloc( n * sizeof( particle ) );
    new_particles = particles;

    float weightsum[n];
    weightsum[0] = particles[0].w;
    for( i = 1; i <n ; i++ )
        weightsum[i] = weightsum[i-1] + particles[i].w;

    float randp[n+1];
    float randtotal = 0;
    for (i=0;i<=n;i++){
        randp[i] = rand() / (float)(RAND_MAX);
        randtotal += randp[i];
    }

    for (i=0;i<=n;i++)
        randp[i] = randp[i] / randtotal;

    float randsum[n+1];
    randsum[0] = randp[0];
    for( i = 1; i <=n ; i++ ){
        randsum[i] = randsum[i-1] + randp[i];
    }

    i=0; // index of randsum
    j=0; // index of weightsum

    while (i<n && j<n) {
        if (weightsum[j]>=randsum[i]){
            new_particles[i] = particles[j];
            new_particles[i].alive = 1;
            new_particles[i].w = 0;
            i++;
        }else{
            j++;
        }
    }

    return new_particles;
}

particle* resample3( particle* particles, int n ,int num_particles)
{
    //    srand(time(0));
    particle* new_particles;
    int i, j = 0;
    float temp;
    new_particles = malloc( n * sizeof( particle ) );
    new_particles = particles;

    float weightsum[n+1];
    weightsum[0] = 0;
    for( i = 1; i <=n ; i++ )
        weightsum[i] = weightsum[i-1] + particles[i-1].w;
    i=0;
    while (i<n) {
        temp =rand() / (float)(RAND_MAX);
        for (j=1; j<=n; j++) {
            if (temp<weightsum[j]){
                new_particles[i] = particles[j-1];
                break;
            }
        }
        i++;
    }
    //    for (i=n;i<num_particles;i++){
    //        new_particles[i].w = 0;
    //    }
    return new_particles;
}




/*
 Compare two particles based on weight.  For use in qsort.

 @param p1 pointer to a particle
 @param p2 pointer to a particle

 @return Returns -1 if the \a p1 has lower weight than \a p2, 1 if \a p1
 has higher weight than \a p2, and 0 if their weights are equal.
 */
int particle_cmp( const void* p1, const void* p2 )
{
    particle* _p1 = (particle*)p1;
    particle* _p2 = (particle*)p2;

    if( _p1->w > _p2->w )
        return -1;
    if( _p1->w < _p2->w )
        return 1;
    return 0;
}

int particle_cmp2( const void* p1, const void* p2 )
{
    particle* _p1 = (particle*)p1;
    particle* _p2 = (particle*)p2;

    if( _p1->alive > _p2->alive )
        return -1;
    if( _p1->alive < _p2->alive )
        return 1;
    return 0;
}



//display_particle

/*
 Displays a particle on an image as a rectangle around the region specified
 by the particle

 @param img the image on which to display the particle
 @param p the particle to be displayed
 @param color the color in which \a p is to be displayed
 */
//void display_particle( Mat mRgb, particle p, CvScalar color )
//{
//    if (p.alive ==1){
//
//  int x0, y0, x1, y1,xcenter,ycenter;
//
//  x0 = round( p.x - 0.5 * p.s * p.width );
//  y0 = round( p.y - 0.5 * p.s * p.height );
//  x1 = x0 + round( p.s * p.width );
//  y1 = y0 + round( p.s * p.height );
//
//    xcenter = (x1 - x0)/2 -1;
//    ycenter = (y1 - y0)/2 -1;
//
//  //cvRectangle( img, cvPoint( x0, y0 ), cvPoint( x1, y1 ), color, 1, 8, 0 );
////    cvRectangle( img, cvPoint( x0+xcenter, y0+ycenter), cvPoint( x1-xcenter, y1-ycenter ), color, 1, 8, 0 );
//    //cvRectangle( img, cvPoint( xcenter-1, ycenter-1 ), cvPoint( ycenter+1, ycenter+1 ), color, 0.5, 8, 0 );
//        Rectangle( mRgb, Point( x0+xcenter, y0+ycenter), Point( x1-xcenter, y1-ycenter ), color, 1, 8, 0 );
//    }
//}

int calculate_alive(particle* particles,int n){
    int num=0;
    int i;
    for (i=0;i<n;i++)
        if (particles[i].alive ==1)
            num++;
    return num;
}

//mean_shift

float euclidean_distance(particle p1, particle p2){
    float total = 0;
    total += (p1.x - p2.x) * (p1.x - p2.x);
    total += (p1.y - p2.y) * (p1.y - p2.y);
    return sqrt(total);
}

float gaussian_kernel(float distance, float kernel_bandwidth,float scale){
    float temp;
    temp =  exp(-(distance*distance/scale) / (kernel_bandwidth));
    return temp;
}

particle Meanshift_cluster( particle* particles, int n, float kernel_bandwidth,int framewidth, int frameheight){
    //n: size of particles

    particle center_particle = particles[0];
    particle prev_center_p;
    float meanshift_distance;
    float total_weight;
    float xshift,yshift;
    int i;
    int iter = 0;
    float s = 0,ww = 0,hh = 0;
    int nums = 10;
    center_particle.s = 1;


    do{
        iter ++;
        prev_center_p = center_particle;
        meanshift_distance = EPSILON + 1.0;
        xshift = 0.0;
        yshift = 0.0;
        total_weight = 0.0;
        for(i = 0;i < n; i++){
            float distance = euclidean_distance(particles[i],center_particle);
            float weight = gaussian_kernel(distance,kernel_bandwidth,100);
            xshift += weight*(particles[i].x - center_particle.x);
            yshift += weight*(particles[i].y - center_particle.y);
            total_weight += weight;
        }
        xshift /= total_weight;
        yshift /= total_weight;

        center_particle.x += xshift;
        center_particle.y += yshift;

        meanshift_distance = euclidean_distance(prev_center_p,center_particle);

    }while (iter<500 && meanshift_distance > EPSILON);

    iter = 0;
    ww = particles[0].width * particles[0].s;
    float wshift;
    float prev_ww;
    do{
        iter ++;
        prev_ww = ww;
        meanshift_distance = 0.01 + 1.0;
        total_weight = 0.0;
        wshift = 0.0;
        for(i = 0;i < n; i++){
            float distance = fabsf(particles[i].width * particles[i].s - ww);
            float weight = gaussian_kernel(distance,0.5,1);
            wshift += weight*(particles[i].width  * particles[i].s - ww);
            total_weight += weight;
        }
        wshift /= total_weight;
        ww += wshift;

        meanshift_distance = fabsf(ww - prev_ww);

    }while (iter < 500 && meanshift_distance > 0.01);
    center_particle.width = ww;
    center_particle.height = ww;
    return center_particle;
}

void getHeatMapColor(float value, float *red, float *green, float *blue)
{
    const int NUM_COLORS = 4;
    static float color[NUM_COLORS][3] = { {0,0,1}, {0,1,0}, {1,1,0}, {1,0,0} };
    // A static array of 4 colors:  (blue,   green,  yellow,  red) using {r,g,b} for each.

    int idx1;        // |-- Our desired color will be between these two indexes in "color".
    int idx2;        // |
    float fractBetween = 0;  // Fraction between "idx1" and "idx2" where our value is.

    if(value <= 0)      {  idx1 = idx2 = 0;            }    // accounts for an input <=0
    else if(value >= 1)  {  idx1 = idx2 = NUM_COLORS-1; }    // accounts for an input >=0
    else
    {
        value = value * (NUM_COLORS-1);        // Will multiply value by 3.
        idx1  = floor(value);                  // Our desired color will be after this index.
        idx2  = idx1+1;                        // ... and before this index (inclusive).
        fractBetween = value - (float)(idx1);    // Distance between the two indexes (0-1).
    }

    *red   = (color[idx2][0] - color[idx1][0])*fractBetween + color[idx1][0];
    *green = (color[idx2][1] - color[idx1][1])*fractBetween + color[idx1][1];
    *blue  = (color[idx2][2] - color[idx1][2])*fractBetween + color[idx1][2];
}

void visualize_particle_heatmap(IplImage* frame, particle* particles, int num_particles, int visualize_num_intervals, int num_alives)
{
    int frame_width = frame->width;
    int frame_height = frame->height;
    int x_index,y_index,num_x,num_y;
    int w_interval,h_interval;
    float max_hist = 0;
    float r,g,b;

    w_interval = frame_width / visualize_num_intervals;
    h_interval = frame_height / visualize_num_intervals;
    num_x = ((float)frame_width / w_interval - (int) frame_width / w_interval > 0)? ((int) frame_width / w_interval +1) :((int) frame_width / w_interval);
    num_y = ((float)frame_height / h_interval - (int) frame_height / h_interval > 0)? ((int) frame_height / h_interval +1) :((int) frame_height / h_interval);
    float heatmap_hist[num_x][num_y];
//    memset(heatmap_hist, 0, num_x * num_y);

    for (int i=0; i < num_x ; i++){
        for (int j=0; j < num_y; j++){
            heatmap_hist[i][j] = 0;
        }
    }
    for (int i = 0; i < num_particles; i++){
        if (particles[i].alive){
            x_index = (int) particles[i].x / w_interval;
            y_index = (int) particles[i].y / h_interval;
            if (x_index > num_x || y_index > num_y || x_index < 0 || y_index < 0)
                continue;
            heatmap_hist[x_index][y_index] = (float) heatmap_hist[x_index][y_index] + 1.0;
            if (heatmap_hist[x_index][y_index] > max_hist){
                max_hist = heatmap_hist[x_index][y_index];
            }
        }
    }

    for (int i = 0; i < num_x; i++){
      for (int j = 0; j < num_y; j++){
          heatmap_hist[i][j] = (float)heatmap_hist[i][j] / max_hist;
      }
    }

    for (int i = 0; i < num_x - 1; i++){
        for (int j = 0; j < num_y - 1; j++){
            getHeatMapColor(heatmap_hist[i][j], &r, &g, &b);
            IplImage * rec=cvCreateImage(cvSize(w_interval,h_interval),frame->depth,frame->nChannels);
            cvRectangle(rec,cvPoint(0,0),cvPoint(w_interval,h_interval),CV_RGB(r*255,g*255,b*255),-1,1,0);
            cvSetImageROI(frame,cvRect(i*w_interval,j*h_interval,w_interval,h_interval));
            cvAddWeighted(frame,0.5,rec,1-0.5,0.0,frame);
        }
    }
    cvResetImageROI(frame);

}


void visualize_particle_heatmap2(IplImage* frame, particle* particles, int num_particles, int visualize_intervals2, int num_alives)
{
    int frame_width = frame->width;
    int frame_height = frame->height;
    float max_hist = 0;
    float r,g,b;
    float heatmap_hist[frame_width][frame_height];
//    memset(heatmap_hist, 0, frame_width * frame_height);

    for (int i=0; i < frame_width ; i++){
        for (int j=0; j < frame_height; j++){
            heatmap_hist[i][j] = 0.0000001;
        }
    }
    float dis = 0;
    float te = 0;

    for (int i = 0; i < num_particles; i++){
        if (particles[i].alive){
            int tempx = particles[i].x;
            int tempy = particles[i].y;
            for (int j = MAX(tempx - particles[i].width * particles[i].s / 2,0); j <= MIN(tempx + particles[i].width * particles[i].s /2 , frame_width - 1);j++){
                for (int k = MAX(tempy -particles[i].height * particles[i].s /2,0); k <= MIN( tempy + particles[i].height * particles[i].s /2,frame_height);k++){
                    if (heatmap_hist[j][k] < 10000){
                        dis = (tempx - j) * (tempx - j) + (tempy - k) * (tempy - k);
                        te = MIN(MAX(gaussian_kernel(dis, particles[i].width * particles[i].s *10 , 1000),0.0000001),1);
                        heatmap_hist[j][k] += te;
                        if (heatmap_hist[j][k] > max_hist){
//                        printf("%f %f\n",heatmap_hist[j][k],max_hist);
                            max_hist = heatmap_hist[j][k];
                        }
                    }
                }
            }
        }
    }
    
//    for (int i = 0; i < num_particles; i++){
//        if (particles[i].alive){
//            int tempx = particles[i].x;
//            int tempy = particles[i].y;
//            for (int j = -particles[i].width * particles[i].s / 4; j<=particles[i].width * particles[i].s /4 ;j++){
//                for (int k = -particles[i].height * particles[i].s /4; k<=particles[i].height * particles[i].s /4;k++){
//                    if (tempx + j < frame_width && tempy + k < frame_height && tempx + j >= 0 && tempy + k >= 0)
//                    {
//                        heatmap_hist[tempx + j][tempy + k] ++;
//                        if (heatmap_hist[tempx + j][tempy + k] > max_hist){
//                            max_hist = heatmap_hist[tempx + j][tempy + k];
//                        }
//                    }
//                }
//            }
//        }
//    }

    for (int i = 0; i < frame_width; i++){
      for (int j = 0; j < frame_height; j++){
          heatmap_hist[i][j] = MIN(MAX((float)heatmap_hist[i][j] / max_hist,0.0001),1);
      }
    }

    IplImage * rec=cvCreateImage(cvSize(frame_width,frame_height),frame->depth,frame->nChannels);

    for (int i = 0; i < frame_width - 1; i++){
        for (int j = 0; j < frame_height - 1; j++){
            getHeatMapColor(heatmap_hist[i][j], &r, &g, &b);
            ((uchar *)(rec->imageData + j*rec->widthStep))[i*rec->nChannels + 0]=b*255; // B
            ((uchar *)(rec->imageData + j*rec->widthStep))[i*rec->nChannels + 1]=g*255; // G
            ((uchar *)(rec->imageData + j*rec->widthStep))[i*rec->nChannels + 2]=r*255; // R
        }
    }

    // cvSetImageROI(frame,cvRect(i*w_interval,j*h_interval,w_interval,h_interval));
    cvAddWeighted(frame,0.7,rec,1-0.7,0.0,frame);
    cvResetImageROI(frame);

}
