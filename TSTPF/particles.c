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

#define TRANS_X_STD /*5.0*/ 200
#define TRANS_Y_STD /*2.5*/ 200
#define X_init_STD /*5.0*/ 50
#define Y_init_STD /*2.5*/ 50
#define TRANS_S_STD 0.05
#define EPSILON 2
#define B0  1.0000

particle* init_distribution( histogram** histos, int num_particles, int img_width, int img_height, float U0, estimate estm)
{
    particle* particles;
    float temp;
    int i, k = 0;

    particles = malloc( num_particles * sizeof( particle ) );

    for (i = 0;i < num_particles; i++){

      temp = rand() / (float)(RAND_MAX);

      if (temp<=U0){
            float xx = estm.pos_x[estm.pointer] + B0 * gaussrand(0,X_init_STD);
            float yy = estm.pos_y[estm.pointer] + B0 * gaussrand(0,Y_init_STD);
            particles[k].x =  MAX( 0.0, MIN( (float)img_width - 1.0, xx ) );
            particles[k].y =  MAX( 0.0, MIN( (float)img_height - 1.0, yy ) );
            particles[k].v_x = estm.velocity_x;
            particles[k].v_y = estm.velocity_y;
            particles[k].av_x = estm.accelerate_vx;
            particles[k].av_y = estm.accelerate_vy;
            particles[k].sw = particles[k].sh = 1.0;
            particles[k].width = estm.width;
            particles[k].height = estm.height;
            particles[k].histo = histos[0];
            particles[k].alive = 1;
            particles[k++].w = 0;
      }
      else{
          particles[k].x = particles[k].y = 0.0;
          particles[k].v_x = particles[k].v_y = 0.0;
          particles[k].av_x = particles[k].av_y = 0.0;
          particles[k].sw = particles[k].sh = 0.0;
          particles[k].width = 0;
          particles[k].height = 0;
          particles[k].histo = histos[0];
          particles[k].alive = 0;
          particles[k++].w = 0;
      }//die

    }

    /* make sure to create exactly p particles */
    while( k < num_particles )
    {
        float xx = estm.pos_x[estm.pointer] + B0 * gaussrand(0,X_init_STD);
        float yy = estm.pos_y[estm.pointer] + B0 * gaussrand(0,Y_init_STD);
        particles[k].x =  MAX( 0.0, MIN( (float)img_width - 1.0, xx ) );
        particles[k].y =  MAX( 0.0, MIN( (float)img_height - 1.0, yy ) );
        particles[k].v_x = estm.velocity_x;
        particles[k].v_y = estm.velocity_y;
        particles[k].av_x = estm.accelerate_vx;
        particles[k].av_y = estm.accelerate_vy;
        particles[k].sw = particles[k].sh = 1.0;
        particles[k].width = estm.width;
        particles[k].height = estm.height;
        particles[k].histo = histos[0];
        particles[k].alive = 1;
        particles[k++].w = 0;
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
particle transition( particle p, int img_width, int img_height,float U0,float U1,histogram** histos, estimate estm, estimate estm_PF)
{
    //    srand(time(0));
    float x, y, s;
    particle pn;

    if (p.alive ==0 && (rand()/(float)(RAND_MAX))<=U0){
        if (estm.valid[estm.pointer] == 1){
            pn.x = MAX( 0.0, MIN( (float)img_width - 1.0, estm.pos_x[estm.pointer]+ B0 * gaussrand(0,X_init_STD) ) );
            pn.y = MAX( 0.0, MIN( (float)img_height - 1.0, estm.pos_y[estm.pointer] + B0 * gaussrand(0,Y_init_STD) ) );
            pn.width = estm.width_record[estm.pointer];
            pn.height = estm.height_record[estm.pointer];
        }else{
            pn.x = rand() % img_width;
            pn.y = rand() % img_height;
            pn.width = estm_PF.width;
            pn.height = estm_PF.height;
        }

        if (estm.istrack){
            pn.v_x = estm.velocity_x;
            pn.v_y = estm.velocity_y;
            pn.av_x = estm.accelerate_vx;
            pn.av_y = estm.accelerate_vy;
        }else{
            pn.v_x = estm_PF.velocity_x;
            pn.v_y = estm_PF.velocity_y;
            pn.av_x = estm_PF.accelerate_vx;
            pn.av_y = estm_PF.accelerate_vy;
        }
        pn.sw = pn.sh = 1.0;
        pn.histo = histos[0];
        pn.alive = 1;
        pn.w = 0;

    }//born
    else if (p.alive == 1 && (rand()/(float)(RAND_MAX))<=1-U1){
        x = p.x + p.v_x + p.av_x + B0 * gaussrand(0, TRANS_X_STD);
        if (x > (float)img_width - 1.0)
            x = x - (float)img_width - 1.0;
        if (x < 0)
            x = (float)img_width - 1.0 -(0 - x);
        pn.x = x;
        y = p.y + p.v_y + p.av_y + B0 * gaussrand(0, TRANS_Y_STD);
        if (y > (float)img_height- 1.0)
            y = y - (float)img_height - 1.0;
        if (y < 0)
            y = (float)img_height - 1.0 - (0 - y);
        pn.y = y;
        s = p.sw + B0 * gaussrand(0, TRANS_S_STD);
        pn.sw = MAX( 0.1, s );
        s = p.sh + B0 * gaussrand(0, TRANS_S_STD);
        pn.sh = MAX( 0.1, s );
        pn.v_x = pn.x - p.x;
        pn.v_y = pn.y - p.y;
        pn.av_x = pn.v_x - p.v_x;
        pn.av_y = pn.v_y - p.v_y;
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
        // pn.sp = pn.s = 0.0;
        pn.sw = pn.sh = 0.0;
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
int normalize_weights( particle* particles, int n )
{
    float sum = 0;
    int i;
    int num_alive = 0;

    for( i = 0; i < n; i++ )
        if (particles[i].alive == 1){
            sum += particles[i].w;
            num_alive ++;
        }

    for( i = 0; i < n; i++ ){
        if (particles[i].alive == 1)
            particles[i].w /= sum;
        else
            particles[i].w =0;
    }
    return num_alive;
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

particle* resample3( particle* particles, int n ,int num_particles)
{
    //    srand(time(0));
    particle* new_particles;
    int i, j = 0;
    float temp;
    new_particles = malloc( num_particles * sizeof( particle ) );
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
        for (i = n;i < num_particles;i++){
            new_particles[i].w = 0;
        }
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

int calculate_alive(particle* particles,int n){
    int num = 0;
    int i;
    for (i = 0; i < n; i ++)
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
    float ww = 0,hh = 0;
    center_particle.sw = 1;
    center_particle.sh = 1;


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

    }while (iter<700 && meanshift_distance > EPSILON);

    iter = 0;
    ww = particles[0].width * particles[0].sw;
    float wshift;
    float prev_ww;
    do{
        iter ++;
        prev_ww = ww;
        meanshift_distance = 0.01 + 1.0;
        total_weight = 0.0;
        wshift = 0.0;
        for(i = 0;i < n; i++){
            float distance = fabsf(particles[i].width * particles[i].sw - ww);
            float weight = gaussian_kernel(distance,0.5,1);
            wshift += weight*(particles[i].width  * particles[i].sw - ww);
            total_weight += weight;
        }
        wshift /= total_weight;
        ww += wshift;

        meanshift_distance = fabsf(ww - prev_ww);

    }while (iter < 700 && meanshift_distance > 0.01);

    iter = 0;
    hh = particles[0].height * particles[0].sh;
    float hshift;
    float prev_hh;
    do{
        iter ++;
        prev_hh = hh;
        meanshift_distance = 0.01 + 1.0;
        total_weight = 0.0;
        hshift = 0.0;
        for(i = 0;i < n; i++){
            float distance = fabsf(particles[i].height * particles[i].sh - hh);
            float weight = gaussian_kernel(distance,0.5,1);
            hshift += weight*(particles[i].height  * particles[i].sh - hh);
            total_weight += weight;
        }
        hshift /= total_weight;
        hh += hshift;

        meanshift_distance = fabsf(hh - prev_hh);

    }while (iter < 700 && meanshift_distance > 0.01);

    center_particle.width = ww;
    center_particle.height = hh;
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
            for (int j = MAX(tempx - particles[i].width * particles[i].sw / 2,0); j <= MIN(tempx + particles[i].width * particles[i].sw /2 , frame_width - 1);j++){
                for (int k = MAX(tempy -particles[i].height * particles[i].sh /2,0); k <= MIN( tempy + particles[i].height * particles[i].sh /2,frame_height);k++){
                    if (heatmap_hist[j][k] < 10000){
                        dis = (tempx - j) * (tempx - j) + (tempy - k) * (tempy - k);
                        te = MIN(MAX(gaussian_kernel(dis, particles[i].width * particles[i].sw *10 , 1000),0.0000001),1);
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
