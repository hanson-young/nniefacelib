#include "config.h"

float pixel_mean[3] = {0, 0, 0};
float pixel_std[3] = {1, 1, 1};
float pixel_scale = 1.0;

float scales32[2] = { 32.f, 16.f };
float scales16[2] = { 8.f, 4.f };
float scales8[2] = { 2.f, 1.f };

float ratios32 = 1.0f;
float ratios16 = 1.0f;
float ratios8 = 1.0f;

anchor_cfg_t anchor_base[] =
{
	{ 32, scales32, ratios32, 16, NULL },
	{ 16, scales16, ratios16, 16, NULL },
	{ 8,  scales8,  ratios8, 16, NULL },
};

float cls_threshold = 0.8;
float nms_threshold = 0.4;
