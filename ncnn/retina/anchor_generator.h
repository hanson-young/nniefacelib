#ifndef ANCHOR_GENERTOR
#define ANCHOR_GENERTOR

#include <vector>
#include <iostream>

#include "config.h"
#include "opencv2/opencv.hpp"
#define LANDMARKS 5
#define MNET_IMG_WIDTH 640
#define MNET_IMG_HEIGHT 640
typedef struct crect2f_s {
	float val[4];
}crect2f_t;

typedef struct anchor_s {
	float anchor[4]; // x1,y1,x2,y2
	float reg[4]; // offset reg
	int center_x; // anchor feat center
	int center_y;
	float score; // cls score
	point_s pts[LANDMARKS]; // pred pts

	rect_t finalbox; // final box res
}anchor_t;

typedef struct anchor_generator_s {
	//需要释放和清空
	//int* anchor_size;
	//float* anchor_ratio;
	//float anchor_step; // scale step

	anchor_t** proposals;
	int proposal_size;
	int pts_count;
	//不需要释放
	int anchor_num; // anchor type num
	crect2f_t** preset_anchors;
	int anchor_stride; // anchor tile stride
	int feature_w; // feature map width
	int feature_h; // feature map height
}anchor_generator_t;

void print_crect2f(crect2f_t* cr);
void ratio_enum(crect2f_t anchor, const float ratios, crect2f_t** ratio_anchors);
void scale_enum(crect2f_t** ratio_anchor, const float* scales, crect2f_t** scale_anchors);
int anchor_init(anchor_generator_t* ag, int stride, const anchor_cfg_t cfg, int dense_anchor);
void landmark_proc(const crect2f_t anchor, const point_t* delta, anchor_t* pts);
void bbox_proc(const crect2f_t anchor, const crect2f_t delta, rect_t* box);
int filter_anchor(ncnn::Mat& cls, ncnn::Mat& reg, ncnn::Mat& pts, anchor_generator_t* ag);
void print_anchor(anchor_t anchor);
void nms(anchor_t* boxes, int total, float threshold, list_t* results);

#endif // ANCHOR_GENERTOR
