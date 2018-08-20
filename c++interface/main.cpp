#include "lib/mask_rcnn.hpp"

int main(int argc, char *argv[])
{
    string model_file = "deploy.prototxt";
    string weights_file = "mask_rcnn.caffemodel";

    const string& video_name = "test_video.mp4";

    Config cfg;
    Video_info video;
    video.video_name;
    cfg.batch_size = atoi(argv[1]);
    cfg.gpu_id = 0;
    Caffe::SetDevice(cfg.gpu_id);
    Caffe::set_mode(Caffe::GPU);
    Detector det = Detector(model_file, weights_file);
    det.cfg = cfg;
    det.video = video;

    det.Detect();

    return 0;
}