//
//  PNClassifier.hpp
//  TSTPF
//
//  Created by BeiXinzhu on 5/10/16.
//  Copyright © 2016 BeiXinzhu. All rights reserved.
//

#ifndef PNClassifier_hpp
#define PNClassifier_hpp

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

cv::Rect TruncateRect(cv::Rect obj_rect, const int img_width, const int img_height);
cv::Rect TruncateRectKeepCenter(const cv::Rect obj_rect, const cv::Size& max_size);
cv::Rect RandomDeformRect(cv::Rect input_rect, const double x_slide_sigma, const double y_slide_sigma, const double aspect_range, cv::RNG& rng);
Mat ImageTransform(IplImage* src, int* region, const double yaw_sigma, const double pitch_sigma, const double roll_sigma,
                   const double blur_max_sigma, const double noise_max_sigma, const double x_slide_sigma, const double y_slide_sigma,
                   const double aspect_range);
void composeExternalMatrix(float yaw, float pitch, float roll, float trans_x, float trans_y, float trans_z, cv::Mat& external_matrix);
Mat Rect2Mat(const cv::Rect img_rect);
void CircumTransImgRect(const cv::Size& img_size, const cv::Mat& transM, cv::Rect_<double>& CircumRect);
void CreateMap(const cv::Size& src_size, const cv::Rect_<double>& dst_rect, const cv::Mat& transMat, cv::Mat& map_x, cv::Mat& map_y);
void RotateImage(const cv::Mat& src, cv::Mat& dst, float yaw, float pitch, float roll,
                 float Z = 1000, int interpolation = cv::INTER_LINEAR, int boarder_mode = cv::BORDER_CONSTANT, const cv::Scalar& border_color = cv::Scalar(0, 0, 0));
cv::Rect ExpandRectForRotate(const cv::Rect area);
void RandomRotateImage(const cv::Mat& src, cv::Mat& dst, float yaw_sigma, float pitch_sigma, float roll_sigma, const cv::Rect area, cv::RNG& rng,float Z = 1000, int interpolation = cv::INTER_LINEAR, int boarder_mode = cv::BORDER_CONSTANT, const cv::Scalar& boarder_color = cv::Scalar(0, 0, 0));

unsigned int generateRandomIndex(unsigned int nTrainSamples, long long int _randomNumber);
float calcInnerProduct(std::vector<float> _weights, float *rowDataPointer, unsigned int _nFeatures);
std::vector<float> updateWeights(std::vector<float> _weights,float innerProduct, float *rowDataPointer, int label, float _learnRate, float _lambda, unsigned int _nFeatures);
std::vector<float> SVMSGD_train(std::vector<float> old_weights, bool onlineUpdate, cv::Mat trainFeatures, cv::Mat labels, float lambda, float learnRate, uint nIterations, float svmthres = 0.0, long long int randomNumber = 1, /*float learnRateDecay = 0.005,*/ float initialize_value = 0.0);
float SVMSGD_predict(std::vector<float> weights, cv::Mat newFeature);


#endif /* PNClassifier_hpp */
