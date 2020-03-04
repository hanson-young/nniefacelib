#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"
#include"anchor_generator.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
int main() {
    extern float pixel_mean[3];
    extern float pixel_std[3];
	std::string param_path =  "./retina/retina.param";
	std::string bin_path = "./retina/retina.bin";
	ncnn::Net _net;
	_net.load_param(param_path.data());
	_net.load_model(bin_path.data());
    cv::Mat src = cv::imread("./retina/41.jpg");
    if(!src.data)
    	printf("load error");

	/*********************  INIT ANCHOR  ***********************/
	anchor_generator_t* anc_gen = NULL;
	anc_gen = (anchor_generator_t*)malloc(sizeof(anchor_generator_t) * FMC);

	for (size_t i = 0; i < FMC; i++)
	{
		printf("anchor base : %d , %f , %f , %f \n", anchor_base[i].stride, anchor_base[i].SCALES[0], anchor_base[i].SCALES[1], anchor_base[i].RATIOS);
		anchor_init(&anc_gen[i], anchor_base[i].stride, anchor_base[i], 0);
		for (int j = 0; j < anc_gen[i].anchor_num; ++j) {
			print_crect2f(anc_gen[i].preset_anchors[j]);
		}
	}

	printf("==================================================\n");
	for (int num = 0; num < 2; num++)
	{
		cv::Mat src;
		if (num % 2)
		{
			src = cv::imread("./retina/41.jpg");

		}
		else
		{
			src = cv::imread("./retina/39.jpg");
		}
		if (!src.data)
			printf("load error");
		ncnn::Mat input = ncnn::Mat::from_pixels_resize(src.data, ncnn::Mat::PIXEL_BGR2RGB, src.cols, src.rows, 384, 672);
		cv::Mat img;
		cv::resize(src, img, cv::Size(384, 672));

		input.substract_mean_normalize(pixel_mean, pixel_std);
		ncnn::Extractor _extractor = _net.create_extractor();
		_extractor.input("data", input);
		/**********************  START  *********************/
		list_t * proposals_l_tmp = list_new();
		for (int i = 0; i < 3; ++i) {
			/**********************  INFERENCE  *********************/
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

			ncnn::Mat out_flatterned = cls.reshape(cls.w * cls.h * cls.c);
			std::vector<float> scores;
			scores.resize(out_flatterned.w);
			for (int j = 0; j<out_flatterned.w; j++)
			{
				scores[j] = out_flatterned[j];
				//printf("%d : %f ,",j, scores[j]);
			}
			/**********************  POST PROCESSING  *********************/
			filter_anchor(cls, reg, pts, &anc_gen[i]);

			for (int r = 0; r < anc_gen[i].pts_count; ++r) {
				list_rpush(proposals_l_tmp, list_node_new(anc_gen[i].proposals[r]));
				//print_anchor(*(anchor_t *)proposals_l_tmp->tail->val);
			}
		}

		for (int r = 0; r < proposals_l_tmp->len; ++r) {
			//proposals[r].print();
			print_anchor(*(anchor_t*)list_at(proposals_l_tmp, r)->val);
		}

		/**********************  NMS FINAL RESULT *********************/
		anchor_t* proposals = NULL;
		proposals = (anchor_t*)malloc(sizeof(anchor_t)*proposals_l_tmp->len);
		for (int i = 0; i < proposals_l_tmp->len; ++i)
		{
			proposals[i] = *(anchor_t *)list_at(proposals_l_tmp, i)->val;
		}
		list_t * results = list_new();
		nms(proposals, proposals_l_tmp->len, nms_threshold, results);

		for (int i = 0; i < results->len; i++)
		{
			anchor_t res = *(anchor_t *)list_at(results, i)->val;
			printf("result rect: %d, %f, %f, %f, %f\n", i, res.finalbox.x1, res.finalbox.y1, res.finalbox.x2, res.finalbox.y2);
			cv::rectangle(img, cv::Point((int)res.finalbox.x1, (int)res.finalbox.y1), cv::Point((int)res.finalbox.x2, (int)res.finalbox.y2), cv::Scalar(0, 255, 255), 2, 8, 0);
			for (int j = 0; j < LANDMARKS; ++j) {
				cv::circle(img, cv::Point((int)res.pts[j].x, (int)res.pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
				printf("result lds: %d, %f, %f\n", j, res.pts[j].x, res.pts[j].y);
			}
		}
		cv::imshow("img", img);
		cv::waitKey(0);
		//free memory
		for (int n = 0; n < FMC; n++)
		{
			for (int i = 0; i < anc_gen[n].proposal_size; i++)
			{
				if (anc_gen[n].proposals[i] != NULL)
				{
					free(anc_gen[n].proposals[i]);
					anc_gen[n].proposals[i] = NULL;
				}
			}
			if (anc_gen[n].proposals != NULL)
			{
				free(anc_gen[n].proposals);
				anc_gen[n].proposals = NULL;
			}
		}
		list_destroy(results);
		list_destroy(proposals_l_tmp);
		if (proposals != NULL)
		{
			free(proposals);
			proposals = NULL;
		}
	}

    return 0;
}

