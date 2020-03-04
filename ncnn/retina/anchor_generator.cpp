#include "anchor_generator.h"

void print_crect2f(crect2f_t* cr) {
	printf("rect %f %f %f %f\n", cr->val[0], cr->val[1], cr->val[2], cr->val[3]);
}

void print_anchor(anchor_t anchor) {
	printf("--finalbox %f %f %f %f, score %f\n", anchor.finalbox.x1, anchor.finalbox.y1, anchor.finalbox.x2, anchor.finalbox.y2, anchor.score);
	printf("--landmarks ");
	int i = 0;
	int size = sizeof(anchor.pts) / sizeof(anchor.pts[0]);
	for (int i = 0; i < size; ++i) {
		printf("%f %f, ", anchor.pts[i].x, anchor.pts[i].y);
	}
	printf("\n");
}

// init different anchors
int anchor_init(anchor_generator_t* ag, int stride, const anchor_cfg_t cfg, int dense_anchor) {
	crect2f_t base_anchor;
	base_anchor.val[0] = 0;
	base_anchor.val[1] = 0;
	base_anchor.val[2] = cfg.BASE_SIZE - 1;
	base_anchor.val[3] = cfg.BASE_SIZE - 1;
	
	//base_size = 16
	crect2f_t** ratio_anchors = NULL;
	(*ag).preset_anchors = NULL;
	// get ratio anchors
	ratio_anchors = (crect2f_t**)malloc(sizeof(crect2f_t*));
	ratio_anchors[0] = (crect2f_t*)malloc(sizeof(crect2f_t));
	ratio_enum(base_anchor, cfg.RATIOS, ratio_anchors);
	//printf("=======%d==========", sizeof(ratio_anchors) / sizeof(ratio_anchors[0]));
	//print_crect2f(ratio_anchors[0]);
	if (ratio_anchors == NULL || *ratio_anchors == NULL)
		printf("ratio_anchors mem alloc error!");

	int ratio_size = sizeof(ratio_anchors) / sizeof(ratio_anchors[0]);
	int scales_size = sizeof(cfg.SCALES) / sizeof(cfg.SCALES[0]);
	
	//printf("ratio_size & scales_size : %d, %d \n", ratio_size, scales_size);
	(*ag).feature_w = (int)(MNET_IMG_WIDTH / cfg.stride);
	(*ag).feature_h = (int)(MNET_IMG_HEIGHT / cfg.stride);
	(*ag).anchor_num = ratio_size * scales_size;
	(*ag).preset_anchors = (crect2f_t**)malloc(sizeof(crect2f_t*) * (*ag).anchor_num);
	int i = 0;
	for (i = 0; i < ratio_size * scales_size; i++)
	{
		(*ag).preset_anchors[i] = (crect2f_t*)malloc(sizeof(crect2f_t));
	}

	scale_enum(ratio_anchors, cfg.SCALES, (*ag).preset_anchors);

	free(*ratio_anchors);
	ratio_anchors = NULL;
	free(ratio_anchors);
	ratio_anchors = NULL;
	printf("\n num : %d \n", (*ag).anchor_num);
	(*ag).anchor_stride = stride;
	return (*ag).anchor_num;
}

int filter_anchor(ncnn::Mat& cls, ncnn::Mat& reg, ncnn::Mat& pts, anchor_generator_t* ag) {
	assert(cls.c == (*ag).anchor_num * 2);
	assert(reg.c == (*ag).anchor_num * 4);
	int pts_length = 0;
	(*ag).proposal_size = 0;
	(*ag).pts_count = 0;
	assert(pts.c % (*ag).anchor_num == 0);
	pts_length = pts.c / (*ag).anchor_num / 2;

	int w = cls.w;
	int h = cls.h;

	int i = 0, j = 0, a = 0, p = 0, id = 0;
	crect2f_t box, delta;
	point_t pts_delta[LANDMARKS];

	for (i = 0; i < h; ++i)
		for (j = 0; j < w; ++j)
			for (a = 0; a < (*ag).anchor_num; ++a)
				if (cls.channel((*ag).anchor_num + a)[i * w + j] >= cls_threshold)
					(*ag).proposal_size++;

	(*ag).proposals = (anchor_t**)malloc(sizeof(anchor_t*) * (*ag).proposal_size);
	for (i = 0; i < (*ag).proposal_size; i++)
	{
		(*ag).proposals[i] = (anchor_t*)malloc(sizeof(anchor_t));
	}

	for (i = 0; i < h; ++i) {
		for (j = 0; j < w; ++j) {
			id = i * w + j;
			for (a = 0; a < (*ag).anchor_num; ++a)
			{ 
				float score = cls.channel((*ag).anchor_num + a)[id];
				//printf("score: %f", score);
				if (cls.channel((*ag).anchor_num + a)[id] >= cls_threshold) {
					
					printf("cls %f\n", cls.channel((*ag).anchor_num + a)[id]);

					box.val[0] = j * (*ag).anchor_stride + (*ag).preset_anchors[a]->val[0];
					box.val[1] = i * (*ag).anchor_stride + (*ag).preset_anchors[a]->val[1];
					box.val[2] = j * (*ag).anchor_stride + (*ag).preset_anchors[a]->val[2];
					box.val[3] = i * (*ag).anchor_stride + (*ag).preset_anchors[a]->val[3];
					printf("%f %f %f %f\n", box.val[0], box.val[1], box.val[2], box.val[3]);

					delta.val[0] = reg.channel(a * 4 + 0)[id];
					delta.val[1] = reg.channel(a * 4 + 1)[id];
					delta.val[2] = reg.channel(a * 4 + 2)[id];
					delta.val[3] = reg.channel(a * 4 + 3)[id];

			
					(*ag).proposals[(*ag).pts_count]->anchor[0] = box.val[0];
					(*ag).proposals[(*ag).pts_count]->anchor[1] = box.val[1];
					(*ag).proposals[(*ag).pts_count]->anchor[2] = box.val[2];
					(*ag).proposals[(*ag).pts_count]->anchor[3] = box.val[3];

					bbox_proc(box, delta, &(*ag).proposals[(*ag).pts_count]->finalbox);
			
					printf("bbox pred\n");
					(*ag).proposals[(*ag).pts_count]->score = cls.channel((*ag).anchor_num + a)[id];
					printf("score: %f", (*ag).proposals[(*ag).pts_count]->score);
					(*ag).proposals[(*ag).pts_count]->center_x = j;
					(*ag).proposals[(*ag).pts_count]->center_y = i;

					printf("center %d %d\n", j, i);
					for (p = 0; p < pts_length; ++p) {
						pts_delta[p].x = pts.channel(a * pts_length * 2 + p * 2)[id];
						pts_delta[p].y = pts.channel(a * pts_length * 2 + p * 2 + 1)[id];
					}
					printf("ready landmark_pred\n");
					landmark_proc(box, pts_delta, &(*(*ag).proposals[(*ag).pts_count]));
					printf("landmark_pred\n");
	
					(*ag).pts_count++;
				}
			}
		}
	}
	printf("+++++++++ %d ++++++++\n", (*ag).pts_count);
	return 0;
}

void ratio_enum(crect2f_t anchor, const float ratios, crect2f_t** ratio_anchors) {
	if (ratio_anchors == NULL || ratio_anchors[0] == NULL)
		printf("ratio_anchors mem malloc error!");
	float w = anchor.val[2] - anchor.val[0] + 1;
	float h = anchor.val[3] - anchor.val[1] + 1;
	float x_ctr = anchor.val[0] + 0.5f * (w - 1);
	float y_ctr = anchor.val[1] + 0.5f * (h - 1);

	float sz = w * h;
	float r = ratios;
	float size_ratios = sz / r;
	float ws = sqrt(size_ratios);
	float hs = ws * r;
	ratio_anchors[0]->val[0] = x_ctr - 0.5f * (ws - 1);
	ratio_anchors[0]->val[1] = y_ctr - 0.5f * (hs - 1);
	ratio_anchors[0]->val[2] = x_ctr + 0.5f * (ws - 1);
	ratio_anchors[0]->val[3] = y_ctr + 0.5f * (hs - 1);
}

void scale_enum(crect2f_t** ratio_anchor, const float* scales, crect2f_t** scale_anchors) {
	int ratio_size = sizeof(ratio_anchor) / sizeof(ratio_anchor[0]);
	int scales_size = sizeof(scales) / sizeof(scales[0]);
	float w, h, x_ctr, y_ctr, ws, hs;
	if (scale_anchors == NULL)
	{
		printf("scale_anchors mem malloc error!");
	}
	for (int a = 0; a < ratio_size; ++a) {
		w = ratio_anchor[a]->val[2] - ratio_anchor[a]->val[0] + 1;
		h = ratio_anchor[a]->val[3] - ratio_anchor[a]->val[1] + 1;
		x_ctr = ratio_anchor[a]->val[0] + 0.5f * (w - 1);
		y_ctr = ratio_anchor[a]->val[1] + 0.5f * (h - 1);

		for (int s = 0; s < scales_size; ++s) {
			ws = w * scales[s];
			hs = h * scales[s];
			scale_anchors[a * ratio_size + s]->val[0] = x_ctr - 0.5f * (ws - 1);
			scale_anchors[a * ratio_size + s]->val[1] = y_ctr - 0.5f * (hs - 1);
			scale_anchors[a * ratio_size + s]->val[2] = x_ctr + 0.5f * (ws - 1);
			scale_anchors[a * ratio_size + s]->val[3] = y_ctr + 0.5f * (hs - 1);
		}
	}
}

void bbox_proc(const crect2f_t anchor, const crect2f_t delta, rect_t* box) {
	float w = anchor.val[2] - anchor.val[0] + 1;
	float h = anchor.val[3] - anchor.val[1] + 1;
	float x_ctr = anchor.val[0] + 0.5 * (w - 1);
	float y_ctr = anchor.val[1] + 0.5 * (h - 1);

	float dx = delta.val[0];
	float dy = delta.val[1];
	float dw = delta.val[2];
	float dh = delta.val[3];

	float pred_ctr_x = dx * w + x_ctr;
	float pred_ctr_y = dy * h + y_ctr;
	float pred_w = exp(dw) * w;
	float pred_h = exp(dh) * h;

	(*box).x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
	(*box).y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
	(*box).x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
	(*box).y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);
}

void landmark_proc(const crect2f_t anchor, const point_t* delta, anchor_t* proposals) {
	float w = anchor.val[2] - anchor.val[0] + 1;
	float h = anchor.val[3] - anchor.val[1] + 1;
	float x_ctr = anchor.val[0] + 0.5 * (w - 1);
	float y_ctr = anchor.val[1] + 0.5 * (h - 1);

	for (int i = 0; i < LANDMARKS; ++i) {
		(*proposals).pts[i].x = delta[i].x * w + x_ctr;
		(*proposals).pts[i].y = delta[i].y * h + y_ctr;
	}
}


int cmp(const void *a, const void *b)
{
	float c = (*(anchor_t*)a).score;
	float d = (*(anchor_t*)b).score;
	return c <= d ? 1 : -1;
}

void nms(anchor_t* boxes, int total, float threshold, list_t* results) {

	int size = total;
	if (size == 0)
		return;
	list_t* idx = list_new();
	list_t* tmp = list_new();
	int tmp_i;
	int count = 0;
	int i = 0;
	int ll[512] = { 0 };
	for (i = 0; i < 512; i++)
	{
		ll[i] = i;
	}
	qsort(boxes, size, sizeof(boxes[0]), cmp);
	for (i = 0; i < size; i++)
	{
		printf("sort:%f", boxes[i].score);
		list_rpush(idx, list_node_new(&ll[i]));
		printf("idx: %d", *(int*)idx->tail->val);
	}
	printf(" size : %d", idx->len);
	while (idx->len > 0)
	{
		int good_idx = *(int*)list_at(idx, 0)->val;
		list_rpush(results, list_node_new(&boxes[good_idx]));
		tmp = list_new();
		for (i = 0; i < idx->len; i++)
		{
			int a = *(int*)(list_at(idx, i)->val);
			list_rpush(tmp, list_node_new(&ll[a]));
		}
		for (i = 0; i < idx->len; i++)
		{
			printf(" tmp : %d", *(int*)list_at(idx, i)->val);
		}
		printf(" size1 : %d", idx->len);
		list_clear(idx);
		printf(" size2 : %d", idx->len);

		for (i = 1; i < tmp->len; i++)
		{
			tmp_i = *(int*)list_at(tmp, i)->val;
			printf("\ntmp_i : %d good_i: %d\n", tmp_i, good_idx);
			printf("x : %f y: %f w:%f h%f\n", boxes[good_idx].finalbox.x1, boxes[good_idx].finalbox.y1, boxes[good_idx].finalbox.x2, boxes[good_idx].finalbox.y2);
			float inter_x1 = MAX(boxes[good_idx].finalbox.x1, boxes[tmp_i].finalbox.x1);
			float inter_y1 = MAX(boxes[good_idx].finalbox.y1, boxes[tmp_i].finalbox.y1);
			float inter_x2 = MIN(boxes[good_idx].finalbox.x2, boxes[tmp_i].finalbox.x2);
			float inter_y2 = MIN(boxes[good_idx].finalbox.y2, boxes[tmp_i].finalbox.y2);

			float w = MAX((inter_x2 - inter_x1 + 1), 0.0F);
			float h = MAX((inter_y2 - inter_y1 + 1), 0.0F);

			float inter_area = w * h;
			float area_1 = (boxes[good_idx].finalbox.y2 - boxes[good_idx].finalbox.y1) * (boxes[good_idx].finalbox.x2 - boxes[good_idx].finalbox.x1);
			float area_2 = (boxes[tmp_i].finalbox.y2 - boxes[tmp_i].finalbox.y1) * (boxes[tmp_i].finalbox.x2 - boxes[tmp_i].finalbox.x1);
			float o = inter_area / (area_1 + area_2 - inter_area);

			if (o <= threshold)
				list_rpush(idx, list_node_new(&ll[tmp_i]));
		}
		list_destroy(tmp);
	}
}


