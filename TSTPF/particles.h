//
//  particles.h
//  TSTPF
//
//  Created by BeiXinzhu on 4/7/16.
//  Copyright © 2016 BeiXinzhu. All rights reserved.
//

#ifndef particles_h
#define particles_h

#include <stdio.h>
//#include "globaldata.hpp"
#include "observation.h"
#include <stdbool.h>
//#include <opencv/highgui.h>
//#include <gsl/gsl_rng.h>

#ifdef __cplusplus
extern "C" {
#endif

    // typedef struct particle {
    //     float x;
    //     /**< current x coordinate */
    //     float y;
    //     /**< current y coordinate */
    //     float s;
    //     /**< scale */
    //     float xp;
    //     /**< previous x coordinate */
    //     float yp;
    //     /**< previous y coordinate */
    //     float sp;
    //     /**< previous scale */
    //     int width;
    //     /**< original width of region described by particle */
    //     int height;
    //     /**< original height of region described by particle */
    //     histogram *histo;
    //     /**< reference histogram describing region being tracked */
    //     float w;
    //     /**< weight */
    //     int alive;
    // } particle;

    typedef struct particle {
        float x;
        /**< current x coordinate */
        float y;
        /**< current y coordinate */
        float sw;
        float sh;
        /**< scale */
        float v_x; // x velocity (pos_x[t] - pos_x[t-1])
        float v_y; // y velocity (pos_y[t] - pos_y[t-1])
        float av_x; // accelerate velocity x (pos_x[t] - pos_x[t-1]) - (pos_x[t-1] - pos_x[t-2])
        float av_y; // accelerate velocity y (pos_y[t] - pos_y[t-1]) - (pos_y[t-1] - pos_y[t-2])
        // float sp;
        // /**< previous scale */
        int width;
        /**< original width of region described by particle */
        int height;
        /**< original height of region described by particle */
        histogram *histo;
        /**< reference histogram describing region being tracked */
        float w;
        /**< weight */
        int alive;
    } particle;


    typedef struct estimate {
        float * pos_x;
        float * pos_y;
        int * width_record;
        int * height_record;
        float confidence;
        float velocity_x;
        float velocity_y;
        float accelerate_vx;
        float accelerate_vy;
        int width;
        int height;
        float * score;
        bool istrack;
    } estimate;

    particle *init_distribution(CvRect *regions, histogram **histos,
                                int n, int p, int w, int h, float U0, float * track_score, int num_match, estimate estm);

    double gaussrand(double E, double V);

    particle transition(particle p, int w, int h, float U0, float U1, CvRect *regions,
                        histogram **histos, float * np, int * xx, int * yy, int * ww, int * hh ,int num_match,estimate estm);

    void normalize_weights(particle *particles, int n);
    particle *resample(particle *particles, int n);

    particle *resample1(particle *particles, int n);

    particle *resample2(particle *particles, int n, int num_particles);

    particle *resample3(particle *particles, int n, int num_particles);
    int particle_cmp(const void *p1, const void *p2);

    int particle_cmp2(const void *p1, const void *p2);
    //void display_particle(Mat mRgb, particle p, CvScalar color);
    int calculate_alive(particle *particles, int n);
    float euclidean_distance(particle p1, particle p2);
    float gaussian_kernel(float distance, float kernel_bandwidth,float scale);
    particle Meanshift_cluster( particle* particles, int n, float kernel_bandwidth,int framewidth, int frameheight);

//    void DrawTransRec(IplImage* img, float ** heatmap_hist, int num_x, int num_y, int w_interval, int h_interval, float alpha);
    void getHeatMapColor(float value, float *red, float *green, float *blue);
    void visualize_particle_heatmap(IplImage* frame, particle* particles, int num_particles, int visualize_num_intervals, int num_alives);
    void visualize_particle_heatmap2(IplImage* frame, particle* particles, int num_particles, int visualize_intervals2, int num_alives);


#ifdef __cplusplus
}
#endif

#endif /* particles_h */
