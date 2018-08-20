#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <time.h>
#include <fstream>
#include <boost/python.hpp>
#include "caffe/caffe.hpp"
#include "gpu_nms.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "mask_rcnn.hpp"
using namespace caffe;
using namespace std;
//using namespace cv;

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  Detector
 *  Description:  Load the model file and weights file
 * =====================================================================================
 */
//load modelfile and weights
Detector::Detector(const string& model_file, const string& weights_file)
{
	net_ = shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
	net_->CopyTrainedLayersFrom(weights_file);
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  Detect
 *  Description:  Perform detection operation
 * =====================================================================================
 */
//perform detection operation
void Detector::Detect()
{
	cv::VideoCapture capture(video.video_name);
	if (!capture.isOpened())
	{
	    cerr << "failed to open " << video.video_name << endl;
	    return;
	}
	video.img_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	video.img_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	cv::Mat* cv_img = new cv::Mat[cfg.batch_size];
	float *im_info = new float[cfg.batch_size*4];
	float *data_buf = new float[cfg.batch_size*cfg.height*cfg.width*3];
	float *boxes = NULL;
	float *pred = NULL;
	float *pred_per_class = NULL;
	float *sorted_pred_cls = NULL;
	int *keep = NULL;
	const float* bbox_delt_all, * pred_cls_all;
	float* bbox_delt, * pred_cls;
	const float* rois;
	float *mask_rois, *input_mask_rois;
	float *mask_kp_rois_all, *det_only_rois_all;
	const float* pred_masks_all, * pred_kps_all;
	float *pred_masks, *xy_preds;
	int num_out, num, num_all, num_mask_kp_box_all, num_det_only_box_all, mask_ind;
    float img_scale_h = cfg.height * 1.0 / video.img_height;
    float img_scale_w = cfg.width * 1.0 / video.img_width;

    int roimask_stage;
    for (int i = 0; i < net_->layer_names().size(); i ++)
    {
        if (net_->layer_names()[i] == "roi_pool6")
        {
            roimask_stage = i;
            break;
        }
    }

    long startTime, endTime, sumTime;
    struct timeval tv_date;

    int frame_id = 0;
    while (true)
    {
        if (!capture.read(cv_img[frame_id % cfg.batch_size]))
		{
			cout << "end reading video " << video.video_name << "at frame " << frame_id << endl;
			break ;
		}
		if(cv_img[frame_id % cfg.batch_size].empty())
        {
            cerr << "Can not get the image file at frame " << frame_id << endl;
            break ;
        }
        cv::Mat cv_resized;
        cv::Mat cv_new(video.img_height, video.img_width, CV_32FC3, cv::Scalar(0,0,0));

        for (int h = 0; h < video.img_height; ++h )
        {
            for (int w = 0; w < video.img_width; ++w)
            {
                cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img[frame_id % cfg.batch_size].at<cv::Vec3b>(cv::Point(w, h))[0])-float(102.9801);
                cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img[frame_id % cfg.batch_size].at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.9465);
                cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img[frame_id % cfg.batch_size].at<cv::Vec3b>(cv::Point(w, h))[2])-float(122.7717);

            }
        }
        cv::resize(cv_new, cv_resized, cv::Size(cfg.width, cfg.height));

        im_info[(frame_id % cfg.batch_size) * 4 + 0] = float(cfg.height);
        im_info[(frame_id % cfg.batch_size) * 4 + 1] = float(cfg.width);
        im_info[(frame_id % cfg.batch_size) * 4 + 2] = img_scale_h;
        im_info[(frame_id % cfg.batch_size) * 4 + 3] = img_scale_w;

        int data_offset = (frame_id % cfg.batch_size) * cfg.height * cfg.width * 3;
        for (int h = 0; h < cfg.height; ++h )
        {
            for (int w = 0; w < cfg.width; ++w)
            {
                data_buf[(0*cfg.height+h)*cfg.width+w + data_offset] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
                data_buf[(1*cfg.height+h)*cfg.width+w + data_offset] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
                data_buf[(2*cfg.height+h)*cfg.width+w + data_offset] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
            }
        }

        if (frame_id % cfg.batch_size == cfg.batch_size - 1)
        {
            input_mask_rois = new float[5];
            gettimeofday(&tv_date, NULL);
            startTime = (long)tv_date.tv_sec * 1000000 + (long)tv_date.tv_usec;
            net_->blob_by_name("data")->Reshape(cfg.batch_size, 3, cfg.height, cfg.width);
            Blob<float> * input_blobs= net_->input_blobs()[0];
            switch(Caffe::mode()){
            case Caffe::CPU:
                memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
                break;
            case Caffe::GPU:
                caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), data_buf, input_blobs->mutable_gpu_data());
                break;
            default:
                LOG(FATAL)<<"Unknow Caffe mode";
            }

            //net_->blob_by_name("data")->set_cpu_data(data_buf);
            net_->blob_by_name("im_info")->Reshape(cfg.batch_size, 4, 1, 1);
            net_->blob_by_name("im_info")->set_cpu_data(im_info);
            net_->blob_by_name("mask_rois")->Reshape(1, 5, 1, 1);
            net_->blob_by_name("mask_rois")->set_cpu_data(input_mask_rois);
            delete []input_mask_rois;

            net_->ForwardTo(roimask_stage - 1);
            cudaDeviceSynchronize();
            gettimeofday(&tv_date, NULL);
            endTime = (long)tv_date.tv_sec * 1000000 + (long)tv_date.tv_usec;
            sumTime = endTime - startTime;
            cout << "detect inference time " << (1.0 * sumTime) / 1000 << endl;

            bbox_delt_all = net_->blob_by_name("bbox_pred")->cpu_data();
            num_all = net_->blob_by_name("rois")->num();

            rois = net_->blob_by_name("rois")->cpu_data();
            pred_cls_all = net_->blob_by_name("cls_prob")->cpu_data();

            num = num_all / cfg.batch_size;
            boxes = new float[num*4];
            pred = new float[num*5*cfg.class_num];
            pred_per_class = new float[num*5];
            sorted_pred_cls = new float[num*5];
            keep = new int[num];
            pred_cls = new float[num*cfg.class_num];
            bbox_delt = new float[num*4*cfg.class_num];
            mask_kp_rois_all = new float[cfg.batch_size*num*7];  //batch_id, x1,y1,x2,y2, score, class_id
            det_only_rois_all = new float[cfg.batch_size*num*7];  //batch_id, x1,y1,x2,y2, score, class_id

            gettimeofday(&tv_date, NULL);
            startTime = (long)tv_date.tv_sec * 1000000 + (long)tv_date.tv_usec;
            num_mask_kp_box_all = 0;
            num_det_only_box_all = 0;
            for (int batch_id = 0; batch_id < cfg.batch_size; batch_id ++)
            {
                for (int i = 0; i < num * cfg.class_num; i ++)
                {
                    pred_cls[i] = pred_cls_all[batch_id * num * cfg.class_num + i];
                }
                for (int n = 0; n < num; n++)
                {
                    if (rois[num * batch_id * 5 + n * 5] == batch_id)
                    {
                        for (int c = 0; c < 4; c++)
                        {
                            boxes[n*4+c] = rois[num * batch_id * 5 + n*5+c+1];
                        }
                    }

                }

                bbox_transform_inv(num, bbox_delt, pred_cls, boxes, pred);

                for (int i = 1; i < cfg.class_num; i ++)
                {
                    for (int j = 0; j < num; j++)
                    {
                        for (int k=0; k<5; k++)
                            pred_per_class[j*5+k] = pred[(i*num+j)*5+k];
                    }
                    boxes_sort(num, pred_per_class, sorted_pred_cls);
                    _nms(keep, &num_out, sorted_pred_cls, num, 5, cfg.nms_thresh, cfg.gpu_id);
                    get_mask_rois(keep, num_out, sorted_pred_cls, mask_kp_rois_all, det_only_rois_all, num_mask_kp_box_all, num_det_only_box_all, batch_id, i);
                }
            }
            gettimeofday(&tv_date, NULL);
            endTime = (long)tv_date.tv_sec * 1000000 + (long)tv_date.tv_usec;
            sumTime = endTime - startTime;
            cout << "detect misc time " << (1.0 * sumTime) / 1000 << endl;

            if (num_mask_kp_box_all == 0)
                continue;

            mask_rois = new float[num_mask_kp_box_all * 5];
            for (int i = 0; i < num_mask_kp_box_all; i ++)
            {
                for (int j = 0; j < 5; j++)
                {
                    mask_rois[i * 5 + j] = mask_kp_rois_all[i * 7 + j];
                }
            }

            gettimeofday(&tv_date, NULL);
            startTime = (long)tv_date.tv_sec * 1000000 + (long)tv_date.tv_usec;
            net_->blob_by_name("mask_rois")->Reshape(num_mask_kp_box_all, 5, 1, 1);
            net_->blob_by_name("mask_rois")->set_cpu_data(mask_rois);

            net_->ForwardFrom(roimask_stage);
            cudaDeviceSynchronize();

            gettimeofday(&tv_date, NULL);
            endTime = (long)tv_date.tv_sec * 1000000 + (long)tv_date.tv_usec;
            sumTime = endTime - startTime;
            cout << "mask kp inference " << (1.0 * sumTime) / 1000 << endl;

            pred_masks_all = net_->blob_by_name("mask_fcn_probs")->cpu_data();
            pred_kps_all = net_->blob_by_name("kps_score")->cpu_data();
            xy_preds = new float[num_mask_kp_box_all*4*cfg.kp_num];

            gettimeofday(&tv_date, NULL);
            startTime = (long)tv_date.tv_sec * 1000000 + (long)tv_date.tv_usec;
            heatmaps_to_keypoints(xy_preds, num_mask_kp_box_all, pred_kps_all, mask_kp_rois_all, img_scale_h, img_scale_w);

            gettimeofday(&tv_date, NULL);
            endTime = (long)tv_date.tv_sec * 1000000 + (long)tv_date.tv_usec;
            sumTime = endTime - startTime;
            cout << "kp misc " << (1.0 * sumTime) / 1000 << endl;

            vis_det_mask_kp(cv_img, num_mask_kp_box_all, num_det_only_box_all, mask_kp_rois_all, det_only_rois_all, pred_masks_all, pred_kps_all, img_scale_h, img_scale_w);


            delete []mask_rois;
            delete []mask_kp_rois_all;
            delete []det_only_rois_all;
            delete []boxes;
            delete []pred;
            delete []pred_per_class;
            delete []keep;
            delete []sorted_pred_cls;
            delete []pred_cls;
            delete []bbox_delt;
            delete []xy_preds;
        }
        frame_id ++;
    }
    delete []cv_img;
    delete []data_buf;
    delete []im_info;
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  heatmap_to_keypoints
 *  Description:  Get kp result
 * =====================================================================================
 */
void Detector::heatmaps_to_keypoints(float* xy_preds, int num_all, const float* pred_kps_all, const float* mask_rois_all, const float img_scale_h, const float img_scale_w)
{
    int x1, y1, x2, y2;
    int box_height, box_width;
    cv::Mat kp_obj(cfg.kp_resolution, cfg.kp_resolution, CV_32FC1);
    cv::Point max_p, min_p;
    double min, max;

    for (int box_id = 0; box_id < num_all; box_id ++)
    {
        x1 = int(mask_rois_all[box_id*7+1]/img_scale_w);
        y1 = int(mask_rois_all[box_id*7+2]/img_scale_h);
        x2 = int(mask_rois_all[box_id*7+3]/img_scale_w);
        y2 = int(mask_rois_all[box_id*7+4]/img_scale_h);

        box_width = x2 - x1 + 1;
        box_height = y2 - y1 + 1;

        for (int kp_cls = 0; kp_cls < cfg.kp_num; kp_cls ++)
        {
            for (int i = 0; i < cfg.kp_resolution; i ++)
                for (int j = 0; j < cfg.kp_resolution; j ++)
                {
                    kp_obj.at<float>(cv::Point(j, i)) = float(pred_kps_all[\
                    box_id * cfg.kp_resolution * cfg.kp_resolution * cfg.kp_num \
                    + kp_cls * cfg.kp_resolution * cfg.kp_resolution
                    + i * cfg.kp_resolution + j]);
                }
                cv::minMaxLoc(kp_obj, &min, &max, &min_p, &max_p);
                xy_preds[box_id*4*cfg.kp_num+cfg.kp_num*0+kp_cls] = max_p.x * box_width / cfg.kp_resolution + x1;
                xy_preds[box_id*4*cfg.kp_num+cfg.kp_num*1+kp_cls] = max_p.y * box_height / cfg.kp_resolution + y1;
                xy_preds[box_id*4*cfg.kp_num+cfg.kp_num*2+kp_cls] = kp_obj.at<float>(cv::Point(max_p.x, max_p.y));
                xy_preds[box_id*4*cfg.kp_num+cfg.kp_num*3+kp_cls] = 0;
        }
    }
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  get_mask_rois
 *  Description:  Get the detection result for mask & kp
 * =====================================================================================
 */
void Detector::get_mask_rois(int* keep, int num_out, float* sorted_pred_cls, float* mask_kp_rois_all, float* det_only_rois_all, int& num_mask_kp, int& num_det_only, const int batch, const int class_id)
{
    int i = 0;
    int mask_kp_num = 0, det_only_num = 0;

    while (i < num_out && sorted_pred_cls[keep[i]*5+4] > cfg.det_thresh)
    {
        if (mask_kp_num + det_only_num >= num_out)
            return;
        if (cfg.class_name[class_id] == "person")
        {
            mask_kp_rois_all[(num_mask_kp + mask_kp_num) * 7 + 0] = batch;
            mask_kp_rois_all[(num_mask_kp + mask_kp_num) * 7 + 1] = sorted_pred_cls[keep[i]*5+0];
            mask_kp_rois_all[(num_mask_kp + mask_kp_num) * 7 + 2] = sorted_pred_cls[keep[i]*5+1];
            mask_kp_rois_all[(num_mask_kp + mask_kp_num) * 7 + 3] = sorted_pred_cls[keep[i]*5+2];
            mask_kp_rois_all[(num_mask_kp + mask_kp_num) * 7 + 4] = sorted_pred_cls[keep[i]*5+3];
            mask_kp_rois_all[(num_mask_kp + mask_kp_num) * 7 + 5] = sorted_pred_cls[keep[i]*5+4];
            mask_kp_rois_all[(num_mask_kp + mask_kp_num) * 7 + 6] = class_id;
            i ++;
            mask_kp_num ++;
        }
        else
        {
            det_only_rois_all[(num_det_only + det_only_num) * 7 + 0] = batch;
            det_only_rois_all[(num_det_only + det_only_num) * 7 + 1] = sorted_pred_cls[keep[i]*5+0];
            det_only_rois_all[(num_det_only + det_only_num) * 7 + 2] = sorted_pred_cls[keep[i]*5+1];
            det_only_rois_all[(num_det_only + det_only_num) * 7 + 3] = sorted_pred_cls[keep[i]*5+2];
            det_only_rois_all[(num_det_only + det_only_num) * 7 + 4] = sorted_pred_cls[keep[i]*5+3];
            det_only_rois_all[(num_det_only + det_only_num) * 7 + 5] = sorted_pred_cls[keep[i]*5+4];
            det_only_rois_all[(num_det_only + det_only_num) * 7 + 6] = class_id;
            i ++;
            det_only_num ++;
        }
    }
    num_mask_kp += mask_kp_num;
    num_det_only += det_only_num;
}


/*
 * ===  FUNCTION  ======================================================================
 *         Name:  vis_det_mask_kp
 *  Description:  Visuallize the detection mask & keypoints result
 * =====================================================================================
 */
void Detector::vis_det_mask_kp(cv::Mat* cv_img, const int num_mask_kp, const int num_det_only, const float* mask_kp_rois_all, const float* det_only_rois_all, const float* pred_masks_all, const float* pred_kps_all, const float img_scale_h, const float img_scale_w)
{
    int box_batch, class_id, x1, y1, x2, y2;
    float score;
	int i1, i2;
	float kps[4][cfg.kp_num];
	cv::Point kp_x, kp_y;
	float mid_shoulder[2], mid_hip[2], sc_mid_shoulder, sc_mid_hip;

	cv::Mat mask_obj(cfg.mask_resolution, cfg.mask_resolution, CV_32FC1);
	cv::Mat mask_obj_resized;

	for (int box_id = 0; box_id < num_det_only; box_id)
	{
		box_batch = int(det_only_rois_all[box_id * 7 + 0]);
		x1 = int(det_only_rois_all[box_id * 7 + 1] / img_scale_w);
		y1 = int(det_only_rois_all[box_id * 7 + 2] / img_scale_h);
		x2 = int(det_only_rois_all[box_id * 7 + 3] / img_scale_w);
		y2 = int(det_only_rois_all[box_id * 7 + 4] / img_scale_h);
		score = det_only_rois_all[box_id * 7 + 5];
		class_id = int(det_only_rois_all[box_id * 7 + 6]);
		if (cfg.show_box)
		{
		    ostringstream stream;
		    stream << box_id << cfg.class_name[class_id] << score;
		    cv::rectangle(cv_img[box_batch],cv::Point(x1, y1),cv::Point(x2, y2),cv::Scalar(255,0,0), 5);
		    cv::putText(cv_img[box_batch],stream.str(),cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255,0,0));
		}
	}
	for (int box_id = 0; box_id < num_mask_kp; box_id)
	{
		box_batch = int(mask_kp_rois_all[box_id * 7 + 0]);
		x1 = int(mask_kp_rois_all[box_id * 7 + 1] / img_scale_w);
		y1 = int(mask_kp_rois_all[box_id * 7 + 2] / img_scale_h);
		x2 = int(mask_kp_rois_all[box_id * 7 + 3] / img_scale_w);
		y2 = int(mask_kp_rois_all[box_id * 7 + 4] / img_scale_h);
		score = mask_kp_rois_all[box_id * 7 + 5];
		class_id = int(mask_kp_rois_all[box_id * 7 + 6]);
		if (cfg.show_box)
		{
		    ostringstream stream;
		    stream << box_id << cfg.class_name[class_id] << score;
		    cv::rectangle(cv_img[box_batch],cv::Point(x1, y1),cv::Point(x2, y2),cv::Scalar(255,0,0), 5);
		    cv::putText(cv_img[box_batch],stream.str(),cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255,0,0));
		}
		if (cfg.show_mask && cfg.class_name[class_id] == "person")
        {
            for (int i = 0; i < cfg.mask_resolution; i ++)
                for (int j = 0; j < cfg.mask_resolution; j ++)
                {
                    mask_obj.at<float>(cv::Point(j, i)) = float(pred_masks_all[\
                        box_id * cfg.mask_resolution * cfg.mask_resolution * cfg.class_num \
                        + class_id * cfg.mask_resolution * cfg.mask_resolution
                        + i * cfg.mask_resolution + j]);
                }
            cv::resize(mask_obj, mask_obj_resized, cv::Size(x2 - x1 + 1, y2 - y1 + 1));

            for (int i = 0; i < y2 - y1 + 1; i ++)
                for (int j = 0; j < x2 - x1 + 1; j ++)
                    mask_obj_resized.at<float>(cv::Point(j, i)) = mask_obj_resized.at<float>(cv::Point(j, i)) > cfg.mask_thresh ? 1 : 0;

            mask_obj_resized.convertTo(mask_obj_resized, CV_8UC1);
            vector<vector<cv::Point> > contours;
            vector<cv::Vec4i> hierarchy;
            cv::findContours(mask_obj_resized, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

            for (int i = 0; i < contours.size(); i ++)
                for (int j = 0; j < contours.size(); j ++)
                {
                    contours[i][j].x += x1;
                    contours[i][j].y += y1;
                }

            for (int i = 0; i < contours.size(); i ++)
            {
                cv::Scalar color = cv::Scalar(255, 255, 255);
                cv::drawContours(cv_img[box_batch], contours, i, color, 5, 8, hierarchy, 0, cv::Point() );
            }
        }
        if (cfg.show_kp && cfg.class_name[class_id] == "person")
        {
            for (int i = 0; i < 4; i ++)
                for (int j = 0; j < cfg.kp_num; j ++)
                    kps[i][j] = pred_kps_all[box_id * cfg.kp_num * 4 + cfg.kp_num * i + j];

            for (int l = 0; l < 15; l ++)
            {
                i1 = cfg.kp_lines[l][0];
                i2 = cfg.kp_lines[l][1];
                if (kps[2][i1] > cfg.kp_thresh && kps[2][i2] > cfg.kp_thresh)
                {
                    kp_x.x = kps[0][i1];
                    kp_x.y = kps[1][i1];
                    kp_y.x = kps[0][i2];
                    kp_y.y = kps[1][i2];
                    cv::line(cv_img[box_batch], kp_x, kp_y, cfg.colors[l], 3);
                }
                if (kps[2][i1] > cfg.kp_thresh)
                {
                    kp_x.x = kps[0][i1];
                    kp_x.y = kps[1][i1];
                    cv::circle(cv_img[box_batch], kp_x, 5, cfg.colors[l], -1);
                }
                if (kps[2][i2] > cfg.kp_thresh)
                {
                    kp_x.x = kps[0][i2];
                    kp_x.y = kps[1][i2];
                    cv::circle(cv_img[box_batch], kp_x, 5, cfg.colors[l], -1);
                }
            }
            mid_shoulder[0] = (kps[0][5] + kps[0][6]) / 2;
            mid_shoulder[1] = (kps[1][5] + kps[1][6]) / 2;
            mid_hip[0] = (kps[0][11] + kps[0][12]) / 2;
            mid_hip[1] = (kps[1][11] + kps[1][12]) / 2;
            sc_mid_shoulder = kps[2][5] < kps[2][6] ? kps[2][5] : kps[2][6];
            sc_mid_hip = kps[2][11] < kps[2][12] ? kps[2][11] : kps[2][12];
            if (sc_mid_shoulder > cfg.kp_thresh && kps[2][0] > cfg.kp_thresh)
            {
                kp_x.x = mid_shoulder[0];
                kp_x.y = mid_shoulder[1];
                kp_y.x = kps[0][0];
                kp_y.y = kps[1][0];
                cv::line(cv_img[box_batch], kp_x, kp_y, cv::Scalar(0, 0, 255), 3);
            }
            if (sc_mid_shoulder > cfg.kp_thresh && sc_mid_hip > cfg.kp_thresh)
            {
                kp_x.x = mid_shoulder[0];
                kp_x.y = mid_shoulder[1];
                kp_y.x = mid_hip[0];
                kp_y.y = mid_hip[1];
                cv::line(cv_img[box_batch], kp_x, kp_y, cv::Scalar(0, 0, 255), 3);
            }
        }
	}
	for (int batch_id = 0; batch_id < cfg.batch_size; batch_id ++)
	{
	    cv::imshow("mask kp video", cv_img[batch_id]);
	    cv::waitKey(5);
	}

}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  boxes_sort
 *  Description:  Sort the bounding box according score
 * =====================================================================================
 */
void Detector::boxes_sort(const int num, const float* pred, float* sorted_pred)
{
	vector<Info> my;
	Info tmp;
	for (int i = 0; i< num; i++)
	{
		tmp.score = pred[i*5 + 4];
		tmp.head = pred + i*5;
		my.push_back(tmp);
	}
	std::sort(my.begin(), my.end(), compare);
	for (int i=0; i<num; i++)
	{
		for (int j=0; j<5; j++)
			sorted_pred[i*5+j] = my[i].head[j];
	}
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  bbox_transform_inv
 *  Description:  Compute bounding box regression value
 * =====================================================================================
 */
void Detector::bbox_transform_inv(int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred)
{
	float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	for(int i=0; i< num; i++)
	{
		width = boxes[i*4+2] - boxes[i*4+0] + 1.0;
		height = boxes[i*4+3] - boxes[i*4+1] + 1.0;
		ctr_x = boxes[i*4+0] + 0.5 * width;
		ctr_y = boxes[i*4+1] + 0.5 * height;
		for (int j=0; j< cfg.class_num; j++)
		{

			dx = box_deltas[(i*cfg.class_num+j)*4+0] / cfg.bbox_reg_weights[0];
			dy = box_deltas[(i*cfg.class_num+j)*4+1] / cfg.bbox_reg_weights[1];
			dw = box_deltas[(i*cfg.class_num+j)*4+2] / cfg.bbox_reg_weights[2];
			dh = box_deltas[(i*cfg.class_num+j)*4+3] / cfg.bbox_reg_weights[3];
			pred_ctr_x = ctr_x + width*dx;
			pred_ctr_y = ctr_y + height*dy;
			pred_w = width * exp(dw);
			pred_h = height * exp(dh);
			pred[(j*num+i)*5+0] = max(min(pred_ctr_x - 0.5* pred_w, video.img_width -1), 0);
			pred[(j*num+i)*5+1] = max(min(pred_ctr_y - 0.5* pred_h, video.img_height -1), 0);
			pred[(j*num+i)*5+2] = max(min(pred_ctr_x + 0.5* pred_w, video.img_width -1), 0);
			pred[(j*num+i)*5+3] = max(min(pred_ctr_y + 0.5* pred_h, video.img_height -1), 0);
			pred[(j*num+i)*5+4] = pred_cls[i*cfg.class_num+j];
		}
	}

}
