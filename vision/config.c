#include "config.h"

float pixel_mean[3] = {0, 0, 0};
float pixel_std[3] = {1, 1, 1};
float pixel_scale = 1.0;

float scales32[SCALES_SIZE] = { 32.f, 16.f };
float scales16[SCALES_SIZE] = { 8.f, 4.f };
float scales8[SCALES_SIZE] = { 2.f, 1.f };

anchor_cfg_t anchor_base[] =
{
	{ 32, scales32, 1.0f, 16, NULL },
	{ 16, scales16, 1.0f, 16, NULL },
	{ 8,  scales8,  1.0f, 16, NULL },
};

float cls_threshold = 0.8 * QUANT_BASE;
float nms_threshold = 0.4;
