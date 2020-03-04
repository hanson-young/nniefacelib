#ifndef FD_CONFIG
#define FD_CONFIG

#include <stdio.h>
#include"list.h"
#include<math.h>
#include "sample_comm_nnie.h"

#define LANDMARKS 5
#define MNET_IMG_WIDTH 640
#define MNET_IMG_HEIGHT 640
#define SCALES_SIZE 2
#define RATIOS_SIZE 1
#define FMC 3
#define QUANT_BASE 4096.0f

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

typedef struct point_s
{
	float x;
	float y;
}point_t;

typedef struct rect_s
{
	float x1;
	float y1;
	float x2;
	float y2;
}rect_t;

typedef struct anchor_cfg_s
{
	int  stride;
	float *SCALES;
	float RATIOS;
	int BASE_SIZE;
	void(*MenuFunc)(void);	

}anchor_cfg_t;

typedef struct crect2f_s {
	float val[4];
}crect2f_t;

typedef struct anchor_s {
	float anchor[4]; // x1,y1,x2,y2
	float reg[4]; // offset reg
	int center_x; // anchor feat center
	int center_y;
	float score; // cls score
	point_t pts[LANDMARKS]; // pred pts

	rect_t finalbox; // final box res
}anchor_t;

typedef struct anchor_generator_s {
	//每一轮循环结束后，需要释放和清空
	//int* anchor_size;
	//float* anchor_ratio;
	//float anchor_step; // scale step
	anchor_t** proposals;
	int proposal_size;
	int pts_count;
	//每一轮循环结束后，不需要释放和清空
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
int filter_anchor(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, int stride_index, anchor_generator_t* ag);
void print_anchor(anchor_t anchor);
void nms(anchor_t* boxes, int total, float threshold, list_t* results);

extern float pixel_mean[3];
extern float pixel_std[3];
extern float pixel_scale;
extern anchor_cfg_t anchor_base[];
extern float cls_threshold;
extern float nms_threshold;
#endif // FD_CONFIG
