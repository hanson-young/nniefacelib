#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"
#include "pfpld.id.h"

int main() {
    extern float pixel_mean[3];
    extern float pixel_std[3];
	std::string param_path =  "../models/retina.param";
	std::string bin_path = "../models/retina.bin";
	std::string pfpld_path = "../models/pfpld.ncnnmodel";
	ncnn::Net _net, pfpld_net;
	_net.load_param(param_path.data());
	_net.load_model(bin_path.data());

	FILE *fp = fopen(pfpld_path.c_str(), "rb");
	if (fp != nullptr) {
		pfpld_net.load_param_bin(fp);
		pfpld_net.load_model(fp);
		fclose(fp);
	}
            
    cv::Mat img = cv::imread("../images/test1.jpg");
    if(!img.data)
    	printf("load error");


	ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, img.cols, img.rows);
//    cv::resize(img, img, cv::Size(300, 300));

    input.substract_mean_normalize(pixel_mean, pixel_std);
	ncnn::Extractor _extractor = _net.create_extractor();
	_extractor.input("data", input);


    std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        ac[i].Init(stride, anchor_cfg[stride], false);
    }

    std::vector<Anchor> proposals;
    proposals.clear();

    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
    	ncnn::Mat cls;
    	ncnn::Mat reg;
    	ncnn::Mat pts;

        // get blob output
        char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
        char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
        _extractor.extract(clsname, cls);
        _extractor.extract(regname, reg);
        _extractor.extract(ptsname, pts);

        printf("cls %d %d %d\n", cls.c, cls.h, cls.w);
        printf("reg %d %d %d\n", reg.c, reg.h, reg.w);
        printf("pts %d %d %d\n", pts.c, pts.h, pts.w);

        ac[i].FilterAnchor(cls, reg, pts, proposals);

        printf("stride %d, res size %d\n", _feat_stride_fpn[i], proposals.size());

        for (int r = 0; r < proposals.size(); ++r) {
            proposals[r].print();
        }
    }

    // nms
    std::vector<Anchor> result;
    nms_cpu(proposals, nms_threshold, result);

    printf("final result %d\n", result.size());
    for(int i = 0; i < result.size(); i ++)
    {
        cv::rectangle (img, cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height), cv::Scalar(255, 255, 0), 2, 8, 0);
//        for (int j = 0; j < result[i].pts.size(); ++j) {
//        	cv::circle(img, cv::Point((int)result[i].pts[j].x, (int)result[i].pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
//        }
        int x1 = (int)result[i].finalbox.x;
        int y1 = (int)result[i].finalbox.y;
        int x2 = (int)result[i].finalbox.width;
        int y2 = (int)result[i].finalbox.height;
		int height = img.rows;
		int width = img.cols;
		int channel = img.channels();
        int w = x2 - x1 + 1;
        int h = y2 - y1 + 1;

		int size_w = (int)(MAX(w, h)*0.9);
		int size_h = (int)(MAX(w, h)*0.9);
		int cx = x1 + w / 2;
		int cy = y1 + h / 2;
		x1 = cx - size_w / 2;
		x2 = x1 + size_w;
		y1 = cy - (int)(size_h * 0.4);
		y2 = y1 + size_h;
		
		int left = 0;
		int top = 0;
		int bottom = 0;
		int right = 0;
		if(x1 < 0)
            left = -x1;
		if (y1 < 0)
            top = -y1;
		if (x1 >= width)
            right = x2 - width;
		if (y1 >= height)
            bottom = y2 - height;
		
		x1 = MAX(0, x1);
		y1 = MAX(0, y1);
		
		x2 = MIN(width, x2);
		y2 = MIN(height, y2);
		
		cv::Mat face_img = img(cv::Rect(x1, y1, x2 - x1, y2 - y1));
		cv::copyMakeBorder(face_img, face_img, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
		

		cv::resize(face_img, face_img, cv::Size(112, 112));
		
		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(
		(unsigned char*)face_img.data,
		ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
		float mean_vals[3] = {0.0, 0.0, 0.0};
        	float norm_vals[3] = {1 / (float)255.0, 1 / (float)255.0, 1 / (float)255.0};
		ncnn_img.substract_mean_normalize(mean_vals, norm_vals);
		
		ncnn::Extractor pfpld_ex = pfpld_net.create_extractor();
		ncnn::Mat pose, landms;
        std::vector<float> angles;
        std::vector<float> landmarks;
		pfpld_ex.input(pfpld_param_id::BLOB_input, ncnn_img);
		pfpld_ex.extract(pfpld_param_id::BLOB_pose, pose);
		pfpld_ex.extract(pfpld_param_id::BLOB_landms, landms);
		for (int j=0; j<pose.w; j++){
            float tmp_angle = pose[j] * 180.0 / CV_PI;
            angles.push_back(tmp_angle);
		}
		
		for (int j=0; j<landms.w / 2; j++)
		{
            float tmp_x = landms[2 * j] * size_w + x1 - left;
            float tmp_y = landms[2 * j + 1] * size_h + y1 -bottom;
            landmarks.push_back(tmp_x);
            landmarks.push_back(tmp_y);
            cv::circle(img, cv::Point((int)tmp_x, (int)tmp_y), 1, cv::Scalar(0,255,0), 1);
		}
		std::cout<<angles[0]<<"  "<<angles[1]<<"  "<<angles[2]<<std::endl;
		plot_pose_cube(img, angles[0], angles[1], angles[2], (int)result[i].pts[2].x, (int)result[i].pts[2].y, w / 2);
    }
    result[0].print();

    cv::imshow("img", img);
    cv::imwrite("../images/result.jpg", img);
    cv::waitKey(0);
    return 0;
}

