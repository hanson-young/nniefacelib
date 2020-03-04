#ifndef FD_CONFIG
#define FD_CONFIG

#include <stdio.h>
#include"net.h"
#include"list.h"
#include"list.h"
#include<math.h>

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

extern float pixel_mean[3];
extern float pixel_std[3];
extern float pixel_scale;

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

#define FMC 3
const int _feat_stride_fpn[3] = { 32, 16, 8 };

extern anchor_cfg_t anchor_base[];
extern bool dense_anchor;
extern float cls_threshold;
extern float nms_threshold;

#endif // FD_CONFIG
