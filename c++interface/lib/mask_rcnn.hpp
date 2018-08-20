#ifndef MASK_RCNN_HPP
#define MASK_RCNN_HPP
#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <boost/python.hpp>
#include "caffe/caffe.hpp"
#include "gpu_nms.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace caffe;
using namespace std;

#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))
//running configs
struct Config{
    int class_num = 3;
    string class_name[3] = {"_background", "person", "face"};
    int batch_size = 16;
    int height = 720;
    int width = 1280;
    int gpu_id = 0;
    float det_thresh = 0.5;
    float nms_thresh = 0.3;
    float bbox_reg_weights[4] = {10.0, 10.0, 5.0, 5.0};
    int mask_resolution = 28;
    int kp_num = 17;
    int kp_resolution = 56;
    float mask_thresh = 0.7;
    int kp_thresh = 4;
    bool show_box = true;
    bool show_mask = true;
    bool show_kp = true;
    //kp drawing params
    int kp_lines[15][2] = {{1, 2}, {1, 0}, {2, 0}, {2, 4}, {1, 3}, {6, 8}, {8, 10}, {5, 7}, {7, 9}, \
    {12, 14}, {14, 16}, {11, 13}, {13, 15}, {6, 5}, {12, 11}};
    cv::Scalar colors[15] = {cv::Scalar(127,0,255), cv::Scalar(95,49,253), cv::Scalar(63,97,250), \
    cv::Scalar(31,142,243), cv::Scalar(0,180,235), cv::Scalar(32,212,224), cv::Scalar(64,236,211),\
    cv::Scalar(96,250,196), cv::Scalar(128,255,179), cv::Scalar(160,256,161), cv::Scalar(192,235,140), \
    cv::Scalar(224,211,119), cv::Scalar(255,178,96), cv::Scalar(255,139,72), cv::Scalar(255,95,48)};
};

//video processing info
struct Video_info{
    string video_name;
    int img_height;
    int img_width;
};


/*
 * ===  Class  ======================================================================
 *         Name:  Detector
 *  Description:  MaskRCNN CXX Detector
 * =====================================================================================
 */
class Detector {
public:
    Config cfg;
    Video_info video;
	Detector(const string& model_file, const string& weights_file);
	void Detect();
    void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred);
	void get_mask_rois(int* keep, int num_out, float* sorted_pred_cls, float* mask_kp_rois_all, float* det_only_rois_all, int& num_mask_kp, int& num_det_only, const int batch, const int class_id);
	void vis_det_mask_kp(cv::Mat* img, const int num_mask_kp, const int num_det_only, const float* mask_kp_rois_all, const float* det_only_rois_all, const float* pred_masks_all, const float* pred_kps_all, const float img_scale_h, const float img_scale_w);
	void boxes_sort(int num, const float* pred, float* sorted_pred);
	void heatmaps_to_keypoints(float* xy_preds, int num_all, const float* pred_kps_all, const float* mask_rois_all, const float img_scale_h, const float img_scale_w);

private:
	shared_ptr<Net<float> > net_;
	Detector(){}
};

//Using for box sort
struct Info
{
	float score;
	const float* head;
};
bool compare(const Info& Info1, const Info& Info2)
{
	return Info1.score > Info2.score;
}

#endif
