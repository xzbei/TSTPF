//
//  PNClassifier.cpp
//  TSTPF
//
//  Created by BeiXinzhu on 5/10/16.
//  Copyright © 2016 BeiXinzhu. All rights reserved.
//

#include "PNClassifier.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
//#include "types.hpp"

using namespace cv;
using namespace std;

cv::Rect TruncateRect(cv::Rect obj_rect, const int img_width, const int img_height){
    cv::Rect resize_rect;
    int x,y,w,h;

    w = obj_rect.x < 0 ? obj_rect.width + obj_rect.x : obj_rect.width;
    x = obj_rect.x < 0 ? 0 : obj_rect.x;
    y = obj_rect.y < 0 ? 0 : obj_rect.y;
    w = w > img_width ? img_width - obj_rect.x : w;
    h = h > img_height ? img_height - obj_rect.y : h;

    resize_rect = cv::Rect(x,y,w,h);

    return resize_rect;
}

cv::Rect TruncateRectKeepCenter(const cv::Rect obj_rect, const cv::Size& max_size)
{
    cv::Rect exp_rect = obj_rect;
    if (exp_rect.x < 0){
        exp_rect.width += 2 * exp_rect.x;
        exp_rect.x = 0;
    }
    if (exp_rect.y < 0){
        exp_rect.height += 2 * exp_rect.y;
        exp_rect.y = 0;
    }
    if (exp_rect.x + exp_rect.width > max_size.width){
        exp_rect.x += (exp_rect.x + exp_rect.width - max_size.width) / 2;
        exp_rect.width = max_size.width - exp_rect.x;
    }
    if (exp_rect.y + exp_rect.height > max_size.height){
        exp_rect.y += (exp_rect.y + exp_rect.height - max_size.height) / 2;
        exp_rect.height = max_size.height - exp_rect.y;
    }
    return exp_rect;
}

cv::Rect RandomDeformRect(cv::Rect input_rect, const double x_slide_sigma, const double y_slide_sigma, const double aspect_range, cv::RNG& rng){
    double x_mv_r = rng.gaussian(x_slide_sigma);
    double y_mv_r = rng.gaussian(y_slide_sigma);
    double aspect_change = rng.gaussian(aspect_range);

    cv::Rect dst_rect;
    double deform = aspect_change / (2.0 + aspect_change);
    dst_rect.width = input_rect.width * (1.0 + deform);
    dst_rect.height = input_rect.height * (1.0 - deform);
    dst_rect.x = input_rect.x + (input_rect.width - dst_rect.width) / 2;
    dst_rect.y = input_rect.y + (input_rect.height - dst_rect.height) / 2;
    dst_rect.x += x_mv_r * dst_rect.width;
    dst_rect.y += y_mv_r * dst_rect.height;

    return dst_rect;
}

void composeExternalMatrix(float yaw, float pitch, float roll, float trans_x, float trans_y, float trans_z, cv::Mat& external_matrix)
{
    external_matrix.release();
    external_matrix.create(3, 4, CV_64FC1);

    double sin_yaw = sin((double)yaw * CV_PI / 180);
    double cos_yaw = cos((double)yaw * CV_PI / 180);
    double sin_pitch = sin((double)pitch * CV_PI / 180);
    double cos_pitch = cos((double)pitch * CV_PI / 180);
    double sin_roll = sin((double)roll * CV_PI / 180);
    double cos_roll = cos((double)roll * CV_PI / 180);

    external_matrix.at<double>(0, 0) = cos_pitch * cos_yaw;
    external_matrix.at<double>(0, 1) = -cos_pitch * sin_yaw;
    external_matrix.at<double>(0, 2) = sin_pitch;
    external_matrix.at<double>(1, 0) = cos_roll * sin_yaw + sin_roll * sin_pitch * cos_yaw;
    external_matrix.at<double>(1, 1) = cos_roll * cos_yaw - sin_roll * sin_pitch * sin_yaw;
    external_matrix.at<double>(1, 2) = -sin_roll * cos_pitch;
    external_matrix.at<double>(2, 0) = sin_roll * sin_yaw - cos_roll * sin_pitch * cos_yaw;
    external_matrix.at<double>(2, 1) = sin_roll * cos_yaw + cos_roll * sin_pitch * sin_yaw;
    external_matrix.at<double>(2, 2) = cos_roll * cos_pitch;

    external_matrix.at<double>(0, 3) = trans_x;
    external_matrix.at<double>(1, 3) = trans_y;
    external_matrix.at<double>(2, 3) = trans_z;
}


Mat Rect2Mat(const cv::Rect img_rect)
{
    cv::Mat srcCoord(3, 4, CV_64FC1);
    srcCoord.at<double>(0, 0) = img_rect.x;
    srcCoord.at<double>(1, 0) = img_rect.y;
    srcCoord.at<double>(2, 0) = 1;
    srcCoord.at<double>(0, 1) = img_rect.x + img_rect.width;
    srcCoord.at<double>(1, 1) = img_rect.y;
    srcCoord.at<double>(2, 1) = 1;
    srcCoord.at<double>(0, 2) = img_rect.x + img_rect.width;
    srcCoord.at<double>(1, 2) = img_rect.y + img_rect.height;
    srcCoord.at<double>(2, 2) = 1;
    srcCoord.at<double>(0, 3) = img_rect.x;
    srcCoord.at<double>(1, 3) = img_rect.y + img_rect.height;
    srcCoord.at<double>(2, 3) = 1;

    return srcCoord;
}

void CircumTransImgRect(const cv::Size& img_size, const cv::Mat& transM, cv::Rect_<double>& CircumRect)
{
    cv::Mat cornersMat = Rect2Mat(cv::Rect(0, 0, img_size.width, img_size.height));

    cv::Mat dstCoord = transM * cornersMat;
    double min_x = std::min(dstCoord.at<double>(0, 0) / dstCoord.at<double>(2, 0), dstCoord.at<double>(0, 3) / dstCoord.at<double>(2, 3));
    double max_x = std::max(dstCoord.at<double>(0, 1) / dstCoord.at<double>(2, 1), dstCoord.at<double>(0, 2) / dstCoord.at<double>(2, 2));
    double min_y = std::min(dstCoord.at<double>(1, 0) / dstCoord.at<double>(2, 0), dstCoord.at<double>(1, 1) / dstCoord.at<double>(2, 1));
    double max_y = std::max(dstCoord.at<double>(1, 2) / dstCoord.at<double>(2, 2), dstCoord.at<double>(1, 3) / dstCoord.at<double>(2, 3));

    CircumRect.x = min_x;
    CircumRect.y = min_y;
    CircumRect.width = max_x - min_x;
    CircumRect.height = max_y - min_y;
}

void CreateMap(const cv::Size& src_size, const cv::Rect_<double>& dst_rect, const cv::Mat& transMat, cv::Mat& map_x, cv::Mat& map_y)
{
    map_x.create(dst_rect.size(), CV_32FC1);
    map_y.create(dst_rect.size(), CV_32FC1);

    double Z = transMat.at<double>(2, 3);

    cv::Mat invTransMat = transMat.inv();
    cv::Mat dst_pos(3, 1, CV_64FC1);
    dst_pos.at<double>(2, 0) = Z;
    for (int dy = 0; dy<map_x.rows; dy++){
        dst_pos.at<double>(1, 0) = dst_rect.y + dy;
        for (int dx = 0; dx<map_x.cols; dx++){
            dst_pos.at<double>(0, 0) = dst_rect.x + dx;
            cv::Mat rMat = -invTransMat(cv::Rect(3, 2, 1, 1)) / (invTransMat(cv::Rect(0, 2, 3, 1)) * dst_pos);
            cv::Mat src_pos = invTransMat(cv::Rect(0, 0, 3, 2)) * dst_pos * rMat + invTransMat(cv::Rect(3, 0, 1, 2));
            map_x.at<float>(dy, dx) = src_pos.at<double>(0, 0) + (float)src_size.width / 2;
            map_y.at<float>(dy, dx) = src_pos.at<double>(1, 0) + (float)src_size.height / 2;
        }
    }
}


void RotateImage(const cv::Mat& src, cv::Mat& dst, float yaw, float pitch, float roll,
                 float Z, int interpolation, int boarder_mode, const cv::Scalar& border_color)
{
    // rotation matrix
    cv::Mat rotMat_3x4;
    composeExternalMatrix(yaw, pitch, roll, 0, 0, Z, rotMat_3x4);

    cv::Mat rotMat = cv::Mat::eye(4, 4, rotMat_3x4.type());
    rotMat_3x4.copyTo(rotMat(cv::Rect(0, 0, 4, 3)));

    // From 2D coordinates to 3D coordinates
    // The center of image is (0,0,0)
    cv::Mat invPerspMat = cv::Mat::zeros(4, 3, CV_64FC1);
    invPerspMat.at<double>(0, 0) = 1;
    invPerspMat.at<double>(1, 1) = 1;
    invPerspMat.at<double>(3, 2) = 1;
    invPerspMat.at<double>(0, 2) = -(double)src.cols / 2;
    invPerspMat.at<double>(1, 2) = -(double)src.rows / 2;

    cv::Mat perspMat = cv::Mat::zeros(3, 4, CV_64FC1);
    perspMat.at<double>(0, 0) = Z;
    perspMat.at<double>(1, 1) = Z;
    perspMat.at<double>(2, 2) = 1;

    cv::Mat transMat = perspMat * rotMat * invPerspMat;
    cv::Rect_<double> CircumRect;
    CircumTransImgRect(src.size(), transMat, CircumRect);

    cv::Mat map_x, map_y;
    CreateMap(src.size(), CircumRect, rotMat, map_x, map_y);
    cv::remap(src, dst, map_x, map_y, interpolation, boarder_mode, border_color);
}


// Keep center and expand rectangle for rotation
cv::Rect ExpandRectForRotate(const cv::Rect area)
{
    cv::Rect exp_rect;

    int w = cvRound(std::sqrt((double)(area.width * area.width + area.height * area.height)));

    exp_rect.width = w;
    exp_rect.height = w;
    exp_rect.x = area.x - (exp_rect.width - area.width) / 2;
    exp_rect.y = area.y - (exp_rect.height - area.height) / 2;

    return exp_rect;
}


void RandomRotateImage(const cv::Mat& src, cv::Mat& dst, float yaw_sigma, float pitch_sigma, float roll_sigma, const cv::Rect area, cv::RNG& rng,float Z, int interpolation, int boarder_mode, const cv::Scalar& boarder_color)
{
    double yaw = rng.gaussian(yaw_sigma);
    double pitch = rng.gaussian(pitch_sigma);
    double roll = rng.gaussian(roll_sigma);
    //double yaw = rng.uniform(-yaw_range / 2, yaw_range / 2);
    //double pitch = rng.uniform(-pitch_range / 2, pitch_range / 2);
    //double roll = rng.uniform(-roll_range / 2, roll_range / 2);

    Rect rect = (area.width <= 0 || area.height <= 0) ? cv::Rect(0, 0, src.cols, src.rows) :
    ExpandRectForRotate(area);
    rect = TruncateRectKeepCenter(rect, src.size());

    cv::Mat rot_img;
    RotateImage(src(rect).clone(), rot_img, yaw, pitch, roll, Z, interpolation, boarder_mode, boarder_color);

    cv::Rect dst_area((rot_img.cols - area.width) / 2, (rot_img.rows - area.height) / 2, area.width, area.height);
    dst_area = TruncateRectKeepCenter(dst_area, rot_img.size());
    dst = rot_img(dst_area).clone();
}

Mat ImageTransform(IplImage * src, int* region, const double yaw_sigma, const double pitch_sigma, const double roll_sigma,const double blur_max_sigma, const double noise_max_sigma, const double x_slide_sigma, const double y_slide_sigma, const double aspect_range){
    cv::RNG rng;
    cv::Rect area = cv::Rect(region[0],region[1],region[2],region[3]);
    // cvSetImageROI(img, area);
    Mat img = cvarrToMat(src);

    // Deform Rect Randomly
    cv::Rect rect = (area.width <= 0 || area.height <= 0) ? cv::Rect(0, 0, img.cols, img.rows) :
    RandomDeformRect(area, x_slide_sigma, y_slide_sigma, aspect_range, rng);
    rect = TruncateRect(rect,img.size().width,img.size().height);

    // Random Rotation
    Mat dst;
    RandomRotateImage(img, dst, yaw_sigma, pitch_sigma, roll_sigma, rect, rng);

    // Random Noise
	double noise_sigma = rng.uniform(0.0, noise_max_sigma);
	if (noise_sigma > 0){
		Mat gauss_noise(dst.size(), CV_32FC(dst.channels()));
		randn(gauss_noise, 0.0, noise_sigma);
		int num = dst.cols * dst.rows * dst.channels();
		unsigned char* dst_ptr = dst.data;
		float* noise_ptr = (float*)gauss_noise.data;
		for (int i = 0; i < num; i++){
			int val = *dst_ptr + *noise_ptr;
			*dst_ptr = (val > 255) ? 255 : (val < 0) ? 0 : val;
			dst_ptr++, noise_ptr++;
		}
	}

    // Random Blur
	Mat dst2;
	double blur_sigma = rng.uniform(0.0, blur_max_sigma);
	int size = blur_sigma * 2.5 + 0.5;
	size += (1 - size % 2);
	if (blur_sigma > 0 && size >= 3){
		Size ksize(size, size);
		GaussianBlur(dst, dst2, ksize, blur_sigma);
	}
	else{
		dst2 = dst;
	}

	return dst2;

}

/* ########################################### SVMSGD ####################################### */


unsigned int generateRandomIndex(unsigned int nTrainSamples, long long int _randomNumber){
	// Choose random sample, using Mikolov's fast almost-uniform random number
    unsigned int _randomIndex;
	_randomNumber = _randomNumber * (unsigned long long) 25214903917 + 11;
	_randomIndex = (unsigned int)(_randomNumber % (unsigned long long) nTrainSamples);
    return _randomIndex;
}

float calcInnerProduct(std::vector<float> _weights, float *rowDataPointer, unsigned int _nFeatures){
	float innerProduct = 0;
	for (uint feat = 0; feat < _nFeatures; ++feat){
		innerProduct += _weights[feat] * rowDataPointer[feat];
	}
	return innerProduct;
}

std::vector<float> updateWeights(std::vector<float> _weights,float innerProduct, float *rowDataPointer, int label, float _learnRate, float _lambda, unsigned int _nFeatures){
	if (label * innerProduct > 1) {
		// Not a support vector, only apply weight decay
		for (uint feat = 0; feat < _nFeatures; feat++) {
			_weights[feat] -= _learnRate * _lambda * _weights[feat];
		}
	} else {
		// It's a support vector, add it to the weights
		for (uint feat = 0; feat < _nFeatures; feat++) {
			_weights[feat] -= _learnRate * (_lambda * _weights[feat] - label * rowDataPointer[feat]);
		}
	}

    return _weights;
}

std::vector<float> SVMSGD_train(std::vector<float> old_weights, bool onlineUpdate, cv::Mat trainFeatures, cv::Mat labels, float lambda, float learnRate, uint nIterations, float svmthres, long long int randomNumber, /*float learnRateDecay = 0.005,*/ float initialize_value){
    //float lambda = 0.000001, float learnRate = 2, uint nIterations = 100000
    unsigned int nFeatures = trainFeatures.cols;
    unsigned int nTrainSamples = trainFeatures.rows;
    unsigned int randomIndex;

    //initialize weights
    std::vector<float> _weights;
    if (onlineUpdate){
        _weights = old_weights;
    }else{
        _weights.reserve(nFeatures);
		for (uint feat = 0; feat < nFeatures; ++feat){
			_weights.push_back(initialize_value);
		}
    }

    float innerProduct;
	// Stochastic gradient descent SVM
	for (uint iter = 0; iter < nIterations; ++iter){
		randomIndex = generateRandomIndex(nTrainSamples, randomNumber);
		innerProduct = calcInnerProduct(_weights,trainFeatures.ptr<float>(randomIndex),nFeatures);
		int label = (labels.at<int>(randomIndex,0) > svmthres) ? 1 : -1; // ensure that labels are -1 or 1
        _weights = updateWeights(_weights,innerProduct, trainFeatures.ptr<float>(randomIndex), label, learnRate, lambda, nFeatures);
	}

    return _weights;
}


float SVMSGD_predict(std::vector<float> weights, cv::Mat newFeature){
	float innerProduct;
	innerProduct = calcInnerProduct(weights, newFeature.ptr<float>(0), newFeature.cols);

	// Resultant label (-1 or 1)
	// int label = (innerProduct>=0) ? 1 : -1;

	return innerProduct;
}


/* ########################################### HOG Descriptor ####################################### */

std::vector<float> Get_HOG_DES(Mat img, Size dsize = Size(80,72), Size blockSize = Size(8,8), Size blockStride = Size(4,4), Size cellSize = Size(4,4), int nbins = 9){
    Mat dst, dst_gray;
    dst = img;
    resize(dst, dst, Size(80,72) );
    if (dst.channels() > 3)
        cvtColor(dst, dst_gray, CV_RGB2GRAY);

    //extract feature
    HOGDescriptor d( dsize, blockSize, blockStride, cellSize, nbins);
    vector< float> descriptorsValues;
    vector< Point> locations;
    d.compute( dst_gray, descriptorsValues, Size(0,0), Size(0,0), locations);

    return descriptorsValues;
}
