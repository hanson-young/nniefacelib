/**
 * FileName:     nnie_face_api.c
 * @author:      Hanson
 * @version      V1.0 
 * Createdate:   
 *
 * Modification  History:
 * Date         Author        Version        Discription
 *
 * Why & What is modified: <修改原因描述>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <math.h>
#include "config.h"
#include "hi_common.h"
#include "hi_comm_sys.h"
#include "hi_comm_svp.h"
#include "sample_comm.h"
#include "sample_comm_svp.h"
#include "sample_comm_nnie.h"
#include "nnie_face_api.h"
#include "sample_svp_nnie_software.h"
#include "sample_comm_ive.h"
#include "hi_type.h"

#include"list.h"

#define SAMPLE_PI 3.1415926535897932384626433832795
/*cnn para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stDetModel = {0};
static SAMPLE_SVP_NNIE_MODEL_S s_stExtModel = {0};
static SAMPLE_SVP_NNIE_MODEL_S s_stPfpldModel = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stDetNnieParam = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stExtNnieParam = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stPfpldNnieParam = {0};

static anchor_generator_t* anc_gen = NULL;
int IsDebugLog = 0;
SAMPLE_SVP_NNIE_CFG_S   stNnieCfg = {0};
HI_S32 as32ResultDet[200 * 15] = { 0 };
HI_S32 u32ResultDetCnt = 0;
int IndexBuffer[512] = { 0 };
#ifdef SAMPLE_SVP_NNIE_PERF_STAT
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_CLREAR()  memset(&s_stOpForwardPerfTmp,0,sizeof(s_stOpForwardPerfTmp));
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_SRC_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_GET_DIFF_TIME(s_stOpForwardPerfTmp.u64SrcFlushTime)
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_PRE_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_GET_DIFF_TIME(s_stOpForwardPerfTmp.u64PreDstFulshTime)
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_AFTER_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_GET_DIFF_TIME(s_stOpForwardPerfTmp.u64AferDstFulshTime)
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_OP_TIME() SAMPLE_SVP_NNIE_PERF_STAT_GET_DIFF_TIME(s_stOpForwardPerfTmp.u64OPTime)


static SAMPLE_SVP_NNIE_OP_PERF_STAT_S   s_stOpForwardPerfTmp = {0};
#else

#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_CLREAR()
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_OP_TIME()

#endif

/******************************************************************************
* function : 打印crect2f_t信息
******************************************************************************/
void print_crect2f(crect2f_t* cr) {
    printf("rect %f %f %f %f\n", cr->val[0], cr->val[1], cr->val[2], cr->val[3]);
}

/******************************************************************************
* function : 打印人脸检测回归框和关键点信息
******************************************************************************/
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

/******************************************************************************
* function : ratio初始化
******************************************************************************/
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

/******************************************************************************
* function : scale初始化
******************************************************************************/
void scale_enum(crect2f_t** ratio_anchor, const float* scales, crect2f_t** scale_anchors) {
    float w, h, x_ctr, y_ctr, ws, hs;
    if (scale_anchors == NULL)
    {
        printf("scale_anchors mem malloc error!");
    }
    for (int a = 0; a < RATIOS_SIZE; ++a) {
        w = ratio_anchor[a]->val[2] - ratio_anchor[a]->val[0] + 1;
        h = ratio_anchor[a]->val[3] - ratio_anchor[a]->val[1] + 1;
        x_ctr = ratio_anchor[a]->val[0] + 0.5f * (w - 1);
        y_ctr = ratio_anchor[a]->val[1] + 0.5f * (h - 1);

        for (int s = 0; s < SCALES_SIZE; ++s) {
            ws = w * scales[s];
            hs = h * scales[s];
            scale_anchors[a * RATIOS_SIZE + s]->val[0] = x_ctr - 0.5f * (ws - 1);
            scale_anchors[a * RATIOS_SIZE + s]->val[1] = y_ctr - 0.5f * (hs - 1);
            scale_anchors[a * RATIOS_SIZE + s]->val[2] = x_ctr + 0.5f * (ws - 1);
            scale_anchors[a * RATIOS_SIZE + s]->val[3] = y_ctr + 0.5f * (hs - 1);
        }
    }
}

/******************************************************************************
* function : 根据anchor和预测值计算真实bbox坐标
******************************************************************************/
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

/******************************************************************************
* function : 根据anchor和预测值计算真实landmark坐标
******************************************************************************/
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

/******************************************************************************
* function : qsort排序依据函数
******************************************************************************/
int cmp(const void *a, const void *b)
{
    float c = (*(anchor_t*)a).score;
    float d = (*(anchor_t*)b).score;
    return c <= d ? 1 : -1;
}

/******************************************************************************
* function : 人脸检测的NMS函数
******************************************************************************/
void mnet_nms(anchor_t* boxes, int total, float threshold, list_t* results) {

    int size = total;
    if (size == 0)
        return;
    list_t* idx = list_new();
    list_t* tmp = list_new();
    int tmp_i;
    int i = 0;
    qsort(boxes, size, sizeof(boxes[0]), cmp);
    for (i = 0; i < size; i++)
    {
        if(IsDebugLog)
            printf("sort:%f", boxes[i].score);
        list_rpush(idx, list_node_new(&IndexBuffer[i]));
        if(IsDebugLog)
            printf("idx: %d", *(int*)idx->tail->val);
    }
    if(IsDebugLog)
        printf(" size : %d", idx->len);
    while (idx->len > 0)
    {
        int good_idx = *(int*)list_at(idx, 0)->val;
        list_rpush(results, list_node_new(&boxes[good_idx]));
        tmp = list_new();
        for (i = 0; i < idx->len; i++)
        {
            int a = *(int*)(list_at(idx, i)->val);
            list_rpush(tmp, list_node_new(&IndexBuffer[a]));
        }
        if(IsDebugLog)
            for (i = 0; i < idx->len; i++)
            {
                printf(" tmp : %d", *(int*)list_at(idx, i)->val);
            }
        list_clear(idx);

        for (i = 1; i < tmp->len; i++)
        {
            tmp_i = *(int*)list_at(tmp, i)->val;
            if(IsDebugLog)
            {
                printf("\ntmp_i : %d good_i: %d\n", tmp_i, good_idx);
                printf("x : %f y: %f w:%f h%f\n", boxes[good_idx].finalbox.x1, boxes[good_idx].finalbox.y1, boxes[good_idx].finalbox.x2, boxes[good_idx].finalbox.y2);
            }
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
                list_rpush(idx, list_node_new(&IndexBuffer[tmp_i]));
        }
        list_destroy(tmp);
    }
    list_destroy(idx);
}

/******************************************************************************
* function : 初始化anchor
******************************************************************************/
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

    int ratio_size = RATIOS_SIZE;
    int scales_size = SCALES_SIZE;
    int i = 0;
    //printf("ratio_size & scales_size : %d, %d \n", ratio_size, scales_size);
    (*ag).anchor_num = ratio_size * scales_size;
    (*ag).preset_anchors = (crect2f_t**)malloc(sizeof(crect2f_t*) * (*ag).anchor_num);
    for (i = 0; i < (*ag).anchor_num; i++)
    {
        (*ag).preset_anchors[i] = (crect2f_t*)malloc(sizeof(crect2f_t));
    }

    scale_enum(ratio_anchors, cfg.SCALES, (*ag).preset_anchors);

    free(*ratio_anchors);
    ratio_anchors = NULL;
    free(ratio_anchors);
    ratio_anchors = NULL;
    // printf("\n num : %d \n", (*ag).anchor_num);
    (*ag).anchor_stride = stride;
    return (*ag).anchor_num;
}

/******************************************************************************
* function : 根据anchor和threshold过滤预测结局，可以过滤掉大部分的框
******************************************************************************/
int filter_anchor(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, int stride_index, anchor_generator_t* ag) {

    HI_S32 reg_w = pstNnieParam->astSegData[0].astDst[stride_index + FMC * 0].unShape.stWhc.u32Width;
    HI_S32 reg_h = pstNnieParam->astSegData[0].astDst[stride_index + FMC * 0].unShape.stWhc.u32Height;
    HI_S32 reg_c = pstNnieParam->astSegData[0].astDst[stride_index + FMC * 0].unShape.stWhc.u32Chn;

    HI_S32 pts_w = pstNnieParam->astSegData[0].astDst[stride_index + FMC * 1].unShape.stWhc.u32Width;
    HI_S32 pts_h = pstNnieParam->astSegData[0].astDst[stride_index + FMC * 1].unShape.stWhc.u32Height;
    HI_S32 pts_c = pstNnieParam->astSegData[0].astDst[stride_index + FMC * 1].unShape.stWhc.u32Chn;

    HI_S32 cls_w = pstNnieParam->astSegData[0].astDst[stride_index + FMC * 2].unShape.stWhc.u32Width;
    HI_S32 cls_h = pstNnieParam->astSegData[0].astDst[stride_index + FMC * 2].unShape.stWhc.u32Height;
    HI_S32 cls_c = pstNnieParam->astSegData[0].astDst[stride_index + FMC * 2].unShape.stWhc.u32Chn;

    HI_S32* reg = (HI_S32* )((HI_U8* )pstNnieParam->astSegData[0].astDst[stride_index + FMC * 0].u64VirAddr);
    HI_S32* pts = (HI_S32* )((HI_U8* )pstNnieParam->astSegData[0].astDst[stride_index + FMC * 1].u64VirAddr);
    HI_S32* cls = (HI_S32* )((HI_U8* )pstNnieParam->astSegData[0].astDst[stride_index + FMC * 2].u64VirAddr);
    if(IsDebugLog)
    {
        printf("cls %d %d %d\n", cls_c, cls_h, cls_w);
        printf("reg %d %d %d\n", reg_c, reg_h, reg_w);
        printf("pts %d %d %d\n", pts_c, pts_h, pts_w);
    }


    //assert(cls_c == (*ag).anchor_num * 2);
    //assert(reg_c == (*ag).anchor_num * 4);
    HI_S32 pts_length = 0, proposal_size = 0;
    (*ag).pts_count = 0;
    HI_S32 anchor_num = (*ag).anchor_num;
    //assert(pts_c % (*ag).anchor_num == 0);
    pts_length = pts_c / (*ag).anchor_num / 2;

    HI_S32 i = 0, j = 0, a = 0, p = 0, id = 0, c = 0, h = 0, w = 0;
    crect2f_t box, delta;
    point_t pts_delta[LANDMARKS];
    float score_t = 0.f;

    HI_S32 c_pt = 0, h_pt = 0, w_pt = 0;
    if(IsDebugLog)
        printf("anchor_num %d\n", anchor_num);
    for (c = 0; c < anchor_num; ++c)
    {
        c_pt = (c + anchor_num) * cls_h * cls_w;
        for (h = 0; h < cls_h; ++h)
        {
            h_pt = c_pt + cls_w * h;
            for (w = 0; w < cls_w; ++w)
            {
                score_t = *(cls + (h_pt + w));
                if (score_t >= cls_threshold)
                {
                    proposal_size++;
                }
            }
        }
    }

    // printf("proposal_size %d\n", proposal_size);
    // if(!proposal_size)
    //  return 0;
    (*ag).proposal_size = proposal_size;
    (*ag).proposals = (anchor_t**)malloc(sizeof(anchor_t*) * proposal_size);
    for (i = 0; i < proposal_size; i++)
    {
        (*ag).proposals[i] = (anchor_t*)malloc(sizeof(anchor_t));
    }
    for (i = 0; i < cls_h; ++i) {
        for (j = 0; j < cls_w; ++j) {
            id = i * cls_w + j;
            for (a = 0; a < anchor_num; ++a)
            {
                score_t = *(cls + cls_w * cls_h * (a + anchor_num) + id);
                if (score_t >= cls_threshold) {
                    
                    // printf("cls %f ", score_t);

                    box.val[0] = j * (*ag).anchor_stride + (*ag).preset_anchors[a]->val[0];
                    box.val[1] = i * (*ag).anchor_stride + (*ag).preset_anchors[a]->val[1];
                    box.val[2] = j * (*ag).anchor_stride + (*ag).preset_anchors[a]->val[2];
                    box.val[3] = i * (*ag).anchor_stride + (*ag).preset_anchors[a]->val[3];
                    if(IsDebugLog)
                        printf("box : %f %f %f %f\n", box.val[0], box.val[1], box.val[2], box.val[3]);

                    delta.val[0] = *(reg + cls_w*cls_h*(a * 4 + 0) + id) / 4096.f;
                    delta.val[1] = *(reg + cls_w*cls_h*(a * 4 + 1) + id) / 4096.f;
                    delta.val[2] = *(reg + cls_w*cls_h*(a * 4 + 2) + id) / 4096.f;
                    delta.val[3] = *(reg + cls_w*cls_h*(a * 4 + 3) + id) / 4096.f;
                    if(IsDebugLog)
                        printf("delta : %f %f %f %f\n", delta.val[0], delta.val[1], delta.val[2], delta.val[3]);
            
                    (*ag).proposals[(*ag).pts_count]->anchor[0] = box.val[0];
                    (*ag).proposals[(*ag).pts_count]->anchor[1] = box.val[1];
                    (*ag).proposals[(*ag).pts_count]->anchor[2] = box.val[2];
                    (*ag).proposals[(*ag).pts_count]->anchor[3] = box.val[3];

                    bbox_proc(box, delta, &(*ag).proposals[(*ag).pts_count]->finalbox);
                    if(IsDebugLog)
                        printf("bbox pred\n");
                    (*ag).proposals[(*ag).pts_count]->score = score_t / 4096.f;
                    // printf("score: %f", (*ag).proposals[(*ag).pts_count]->score);
                    (*ag).proposals[(*ag).pts_count]->center_x = j;
                    (*ag).proposals[(*ag).pts_count]->center_y = i;
                    if(IsDebugLog)
                        printf("center %d %d\n", i, j);
                    for (p = 0; p < pts_length; ++p) {
                        pts_delta[p].x = *(pts + cls_w*cls_h*(a * pts_length * 2 + p * 2) + id) / 4096.f;
                        pts_delta[p].y = *(pts + cls_w*cls_h*(a * pts_length * 2 + p * 2 + 1) + id) / 4096.f; 
                    }
                    if(IsDebugLog)
                        printf("ready landmark_pred\n");
                    landmark_proc(box, pts_delta, &(*(*ag).proposals[(*ag).pts_count]));
                    if(IsDebugLog)
                        printf("landmark_pred\n");
    
                    (*ag).pts_count++;
                }
            }
        }
    }
    return 0;
}

/******************************************************************************
* function : NNIE Forward
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Forward(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S* pstInputDataIdx,
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S* pstProcSegIdx,HI_BOOL bInstant)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0, j = 0;
    HI_BOOL bFinish = HI_FALSE;
    SVP_NNIE_HANDLE hSvpNnieHandle = 0;
    HI_U32 u32TotalStepNum = 0;
    SAMPLE_SVP_NIE_PERF_STAT_DEF_VAR();

    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_CLREAR();

    SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
        SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr),
        pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);

    SAMPLE_SVP_NNIE_PERF_STAT_BEGIN();
    for(i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
    {
        if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
        {
            for(j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
            {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep)+j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum*pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);

        }
        else
        {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }
    SAMPLE_SVP_NNIE_PERF_STAT_END();
    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_PRE_DST_FLUSH_TIME();

    /*set input blob according to node name*/
    if(pstInputDataIdx->u32SegIdx != pstProcSegIdx->u32SegIdx)
    {
        for(i = 0; i < pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].u16SrcNum; i++)
        {
            for(j = 0; j < pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum; j++)
            {
                if(0 == strncmp(pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].astDstNode[j].szName,
                    pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].astSrcNode[i].szName,
                    SVP_NNIE_NODE_NAME_LEN))
                {
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc[i] =
                        pstNnieParam->astSegData[pstInputDataIdx->u32SegIdx].astDst[j];
                    break;
                }
            }
            SAMPLE_SVP_CHECK_EXPR_RET((j == pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum),
                HI_FAILURE,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,can't find %d-th seg's %d-th src blob!\n",
                pstProcSegIdx->u32SegIdx,i);
        }
    }

    /*NNIE_Forward*/
    SAMPLE_SVP_NNIE_PERF_STAT_BEGIN();
    s32Ret = HI_MPI_SVP_NNIE_Forward(&hSvpNnieHandle,
        pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc,
        pstNnieParam->pstModel, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst,
        &pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx], bInstant);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_Forward failed!\n");

    if(bInstant)
    {
        /*Wait NNIE finish*/
        while(HI_ERR_SVP_NNIE_QUERY_TIMEOUT == (s32Ret = HI_MPI_SVP_NNIE_Query(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].enNnieId,
            hSvpNnieHandle, &bFinish, HI_TRUE)))
        {
            usleep(100);
            SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_INFO,
                "HI_MPI_SVP_NNIE_Query Query timeout!\n");
        }
    }
    SAMPLE_SVP_NNIE_PERF_STAT_END();
    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_OP_TIME();
    u32TotalStepNum = 0;

    SAMPLE_SVP_NNIE_PERF_STAT_BEGIN();
    for(i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
    {
        if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
        {
            for(j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
            {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep)+j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum*pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);

        }
        else
        {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }
    SAMPLE_SVP_NNIE_PERF_STAT_END();
    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_AFTER_DST_FLUSH_TIME();

    return s32Ret;
}

/******************************************************************************
* function : Fill Src Data
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_FillSrcData(SAMPLE_SVP_NNIE_CFG_S* pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S* pstInputDataIdx)
{
    FILE* fp = NULL;
    HI_U32 i =0, j = 0, n = 0;
    HI_U32 u32Height = 0, u32Width = 0, u32Chn = 0, u32Stride = 0, u32Dim = 0;
    HI_U32 u32VarSize = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U8*pu8PicAddr = NULL;
    HI_U32*pu32StepAddr = NULL;
    HI_U32 u32SegIdx = pstInputDataIdx->u32SegIdx;
    HI_U32 u32NodeIdx = pstInputDataIdx->u32NodeIdx;
    HI_U32 u32TotalStepNum = 0;
    printf("Info, open file!\n");
    /*open file*/
    if (NULL != pstNnieCfg->pszPic)
    {
        fp = fopen(pstNnieCfg->pszPic,"rb");
        SAMPLE_SVP_CHECK_EXPR_RET(NULL == fp,HI_INVALID_VALUE,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error, open file failed!\n");
    }

    /*get data size*/
    if(SVP_BLOB_TYPE_U8 <= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType &&
        SVP_BLOB_TYPE_YVU422SP >= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
    {
        u32VarSize = sizeof(HI_U8);
    }
    else
    {
        u32VarSize = sizeof(HI_U32);
    }

    /*fill src data*/
    if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
    {
        u32Dim = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stSeq.u32Dim;
        u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
        pu32StepAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stSeq.u64VirAddrStep);
        pu8PicAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U8,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
        for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
        {
            for(i = 0;i < *(pu32StepAddr+n); i++)
            {
                s32Ret = fread(pu8PicAddr,u32Dim*u32VarSize,1,fp);
                SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                pu8PicAddr += u32Stride;
            }
            u32TotalStepNum += *(pu32StepAddr+n);
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
            SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr),
            u32TotalStepNum*u32Stride);
    }
    else
    {
        u32Height = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Height;
        u32Width = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Width;
        u32Chn = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Chn;
        u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
        pu8PicAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U8,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
        if(SVP_BLOB_TYPE_YVU420SP== pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
        {
            for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
            {
                for(i = 0; i < u32Chn*u32Height/2; i++)
                {
                    s32Ret = fread(pu8PicAddr,u32Width*u32VarSize,1,fp);
                    SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                    pu8PicAddr += u32Stride;
                }
            }
        }
        else if(SVP_BLOB_TYPE_YVU422SP== pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
        {
            for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
            {
                for(i = 0; i < u32Height*2; i++)
                {
                    s32Ret = fread(pu8PicAddr,u32Width*u32VarSize,1,fp);
                    SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                    pu8PicAddr += u32Stride;
                }
            }
        }
        else
        {
            for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
            {
                for(i = 0;i < u32Chn; i++)
                {
                    for(j = 0; j < u32Height; j++)
                    {
                        s32Ret = fread(pu8PicAddr,u32Width*u32VarSize,1,fp);
                        SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                        pu8PicAddr += u32Stride;
                    }
                }
            }
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
            SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr),
            pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num*u32Chn*u32Height*u32Stride);
    }

    fclose(fp);
    return HI_SUCCESS;
FAIL:

    fclose(fp);
    return HI_FAILURE;
}

/******************************************************************************
* function : print report result
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_PrintReportResult(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam)
{
    HI_U32 u32SegNum = pstNnieParam->pstModel->u32NetSegNum;
    HI_U32 i = 0, j = 0, k = 0, n = 0;
    HI_U32 u32SegIdx = 0, u32NodeIdx = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_CHAR acReportFileName[SAMPLE_SVP_NNIE_REPORT_NAME_LENGTH] = {'\0'};
    FILE* fp = NULL;
    HI_U32*pu32StepAddr = NULL;
    HI_S32*ps32ResultAddr = NULL;
    HI_U32 u32Height = 0, u32Width = 0, u32Chn = 0, u32Stride = 0, u32Dim = 0;

    for(u32SegIdx = 0; u32SegIdx < u32SegNum; u32SegIdx++)
    {
        for(u32NodeIdx = 0; u32NodeIdx < pstNnieParam->pstModel->astSeg[u32SegIdx].u16DstNum; u32NodeIdx++)
        {
            s32Ret = snprintf(acReportFileName,SAMPLE_SVP_NNIE_REPORT_NAME_LENGTH,
                "seg%d_layer%d_output%d_inst.linear.hex",u32SegIdx,
                pstNnieParam->pstModel->astSeg[u32SegIdx].astDstNode[u32NodeIdx].u32NodeId,0);
            SAMPLE_SVP_CHECK_EXPR_RET(s32Ret < 0,HI_INVALID_VALUE,SAMPLE_SVP_ERR_LEVEL_ERROR,
                "Error,create file name failed!\n");

            fp = fopen(acReportFileName,"w");
            SAMPLE_SVP_CHECK_EXPR_RET(NULL == fp,HI_INVALID_VALUE,SAMPLE_SVP_ERR_LEVEL_ERROR,
                "Error,open file failed!\n");

            if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].enType)
            {
                u32Dim = pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].unShape.stSeq.u32Dim;
                u32Stride = pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].u32Stride;
                pu32StepAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].unShape.stSeq.u64VirAddrStep);
                ps32ResultAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32,pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].u64VirAddr);

                for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].u32Num; n++)
                {
                    for(i = 0;i < *(pu32StepAddr+n); i++)
                    {
                        for(j = 0; j < u32Dim; j++)
                        {
                            s32Ret = fprintf(fp ,"%08x\n",*(ps32ResultAddr+j));
                            SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret < 0,PRINT_FAIL,
                                SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,write report result file failed!\n");
                        }
                        ps32ResultAddr += u32Stride/sizeof(HI_U32);
                    }
                }
            }
            else
            {
                u32Height = pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].unShape.stWhc.u32Height;
                u32Width = pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].unShape.stWhc.u32Width;
                u32Chn = pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].unShape.stWhc.u32Chn;
                u32Stride = pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].u32Stride;
                ps32ResultAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32,pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].u64VirAddr);
                for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].u32Num; n++)
                {
                    for(i = 0;i < u32Chn; i++)
                    {
                        for(j = 0; j < u32Height; j++)
                        {
                            for(k = 0; k < u32Width; k++)
                            {
                                s32Ret = fprintf(fp,"%08x\n",*(ps32ResultAddr+k));
                                SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret < 0,PRINT_FAIL,
                                    SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,write report result file failed!\n");
                            }
                            ps32ResultAddr += u32Stride/sizeof(HI_U32);
                        }
                    }
                }
            }
            fclose(fp);
        }
    }
    return HI_SUCCESS;

PRINT_FAIL:
    fclose(fp);
    return HI_FAILURE;
}

/******************************************************************************
* function : Cnn Deinit
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Cnn_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_MODEL_S* pstNnieModel)
{

    HI_S32 s32Ret = HI_SUCCESS;
    /*hardware para deinit*/
    if(pstNnieParam!=NULL)
    {
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
    }
    /*model deinit*/
    if(pstNnieModel!=NULL)
    {
        s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
    }
    return s32Ret;
}

/******************************************************************************
* function : Cnn init
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Cnn_ParamInit(SAMPLE_SVP_NNIE_CFG_S* pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstCnnPara)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /*init hardware para*/
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstNnieCfg,pstCnnPara);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,INIT_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n",s32Ret);

    return s32Ret;
INIT_FAIL_0:
    s32Ret = SAMPLE_SVP_NNIE_Cnn_Deinit(pstCnnPara, NULL);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error(%#x),SAMPLE_SVP_NNIE_Cnn_Deinit failed!\n",s32Ret);
    return HI_FAILURE;

}

static HI_S32 SVP_NNIE_MNET(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam)
{
    

    list_t * proposals_l_tmp = list_new();
    list_t * results = list_new();

    for (int i = 0; i < FMC; ++i) {
        filter_anchor(pstNnieParam, i, &anc_gen[i]);
        //printf("stride %d, res size %d\n", _feat_stride_fpn[i], proposals.size());
        for (int r = 0; r < anc_gen[i].pts_count; ++r) {
            //proposals[r].print();
            list_rpush(proposals_l_tmp, list_node_new(anc_gen[i].proposals[r]));
            //print_anchor(*(anchor_t *)proposals_l_tmp->tail->val);
        }
    }

    // filter_anchor(pstNnieParam, 1, &anc_gen[2]);
    if(IsDebugLog)
        for (int r = 0; r < proposals_l_tmp->len; ++r) {
            //proposals[r].print();
            print_anchor(*(anchor_t*)list_at(proposals_l_tmp, r)->val);
        }
    anchor_t* proposals = NULL;
    proposals = (anchor_t*)malloc(sizeof(anchor_t)*proposals_l_tmp->len);
    for (int i = 0; i < proposals_l_tmp->len; ++i)
    {
        proposals[i] = *(anchor_t *)list_at(proposals_l_tmp, i)->val;
    }
    mnet_nms(proposals, proposals_l_tmp->len, nms_threshold, results);
    memset(as32ResultDet, 0, sizeof(as32ResultDet));
    u32ResultDetCnt = results->len;
    for(int i = 0; i < results->len; i ++)
    {
        anchor_t res = *(anchor_t *)list_at(results, i)->val;
        printf("result rect: %d, %f, %f, %f, %f\n", i ,res.finalbox.x1, res.finalbox.y1, res.finalbox.x2, res.finalbox.y2);
        as32ResultDet[i*15 + 0] = res.finalbox.x1;
        as32ResultDet[i*15 + 1] = res.finalbox.x1;
        as32ResultDet[i*15 + 2] = res.finalbox.y1;
        as32ResultDet[i*15 + 3] = res.finalbox.x2;
        as32ResultDet[i*15 + 4] = res.finalbox.y2;

        for (int j = 0; j < LANDMARKS; ++j) {
            as32ResultDet[i*15 + j * 2 + 5] = res.pts[j].x;
            as32ResultDet[i*15 + j * 2 + 6] = res.pts[j].y;
            printf("result lds: %d, %f, %f\n", j + 1 ,res.pts[j].x, res.pts[j].y);
        }
    }

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
        anc_gen[n].proposal_size = 0;
    }
    list_destroy(results);
    list_destroy(proposals_l_tmp);
    if (proposals != NULL)
    {
        free(proposals);
        proposals = NULL;
    }

}

static HI_S32 FACE_DETECTOR_PARAM_INIT(float threshold, int isLog)
{
    memset(as32ResultDet, 0, sizeof(as32ResultDet));
    u32ResultDetCnt = 0;
    IsDebugLog = isLog;
    cls_threshold = threshold * QUANT_BASE;
    for (int i = 0; i < 512; i++)
    {
        IndexBuffer[i] = i;
    }
    anc_gen = (anchor_generator_t*)malloc(sizeof(anchor_generator_t) * FMC);

    for (HI_S32 i = 0; i < FMC; i++)
    {
        anchor_init(&anc_gen[i], anchor_base[i].stride, anchor_base[i], 0);
        if (IsDebugLog)
        {
            printf("\n anchor base : %d , %f , %f , %f \n", anchor_base[i].stride, anchor_base[i].SCALES[0], anchor_base[i].SCALES[1], anchor_base[i].RATIOS);
            for (int j = 0; j < anc_gen[i].anchor_num; ++j)
                print_crect2f(anc_gen[i].preset_anchors[j]);
        }
    }
}

static void FACE_DETECTOR_PARAM_DEINIT()
{
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
        anc_gen[n].proposal_size = 0;
    }
    if (anc_gen != NULL)
    {
        free(anc_gen);
        anc_gen = NULL;
    }

}

void NNIE_FACE_DETECTOR_INIT(char *pcModelName, float threshold, int isLog)
{
    printf("args : %f, %d\n", threshold, isLog);
    HI_S32 s32Ret = HI_SUCCESS;
    /*Set configuration parameter*/
    HI_U32 u32PicNum = 1;
    /*Set configuration parameter*/
    stNnieCfg.u32MaxInputNum = u32PicNum; //max input image num in each batch
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0;//set NNIE core

    /*Sys init*/
    SAMPLE_COMM_SVP_CheckSysInit();

    /*CNN Load model*/
    SAMPLE_SVP_TRACE_INFO("Cnn Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName,&s_stDetModel);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");
    FACE_DETECTOR_PARAM_INIT(threshold, isLog);
    /*CNN parameter initialization*/
    /*Cnn software parameters are set in SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit,
     if user has changed net struct, please make sure the parameter settings in
     SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit function are correct*/
    SAMPLE_SVP_TRACE_INFO("Cnn parameter initialization!\n");
    s_stDetNnieParam.pstModel = &s_stDetModel.stModel;
    s32Ret = SAMPLE_SVP_NNIE_Cnn_ParamInit(&stNnieCfg,&s_stDetNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Cnn_ParamInit failed!\n");
    SAMPLE_SVP_TRACE_INFO("NNIE AddTskBuf!\n");
    /*record tskBuf*/
    s32Ret = HI_MPI_SVP_NNIE_AddTskBuf(&(s_stDetNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_AddTskBuf failed!\n");
    SAMPLE_SVP_TRACE_INFO("NNIE AddTskBuf end!\n");
    return;

CNN_FAIL_1:
    /*Remove TskBuf*/
    s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(s_stDetNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_RemoveTskBuf failed!\n");

CNN_FAIL_0:
    SAMPLE_SVP_TRACE_INFO("Why \n");
    SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stDetNnieParam, &s_stDetModel);
    SAMPLE_COMM_SVP_CheckSysExit();
}


void NNIE_FACE_DETECTOR_GET(HI_CHAR *pcSrcFile)
{
    stNnieCfg.pszPic= pcSrcFile;
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};

    /*Fill src data*/
    SAMPLE_SVP_TRACE_INFO("Cnn start!\n");
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg,&s_stDetNnieParam,&stInputDataIdx);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_1,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");
    SAMPLE_SVP_TRACE_INFO("Load Img!\n");
    /*NNIE process(process the 0-th segment)*/
    long spend;
    struct timespec start, next, end;
    clock_gettime(0, &start);
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stDetNnieParam,&stInputDataIdx,&stProcSegIdx,HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_1,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");
    SAMPLE_SVP_TRACE_INFO("Forward!\n");
    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    printf("\n[inference]===== TIME SPEND: %ld ms =====\n", spend);
    /*Software process*/
    SVP_NNIE_MNET(&s_stDetNnieParam);
    clock_gettime(0, &next);
    spend = (next.tv_sec - end.tv_sec) * 1000 + (next.tv_nsec - end.tv_nsec) / 1000000;
    printf("\n[post process]===== TIME SPEND: %ld ms =====\n", spend);
    /*Print result*/
    // s32Ret = SAMPLE_SVP_NNIE_PrintReportResult(&s_stDetNnieParam);
    // SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,SAMPLE_SVP_NNIE_PrintReportResult failed!");
    return;
CNN_FAIL_1:
    /*Remove TskBuf*/
    SAMPLE_SVP_TRACE_INFO("Why1 \n");
    s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(s_stDetNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_RemoveTskBuf failed!\n");

CNN_FAIL_0:
    SAMPLE_SVP_TRACE_INFO("Why2 \n");
    SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stDetNnieParam, &s_stDetModel);
    SAMPLE_COMM_SVP_CheckSysExit();
}



/******************************************************************************
* function : Cnn sample signal handle
******************************************************************************/
void NNIE_FACE_DETECTOR_RELEASE(void)
{
    FACE_DETECTOR_PARAM_DEINIT();
    SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stDetNnieParam, &s_stDetModel);
    memset(&s_stDetNnieParam,0,sizeof(SAMPLE_SVP_NNIE_PARAM_S));
    memset(&s_stDetModel,0,sizeof(SAMPLE_SVP_NNIE_MODEL_S));
    SAMPLE_COMM_SVP_CheckSysExit();
}



void NNIE_FACE_EXTRACTOR_INIT(char *pcModelName)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /*Set configuration parameter*/
    HI_U32 u32PicNum = 1;
    /*Set configuration parameter*/
    stNnieCfg.u32MaxInputNum = u32PicNum; //max input image num in each batch
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0;//set NNIE core

    /*Sys init*/
    SAMPLE_COMM_SVP_CheckSysInit();

    /*CNN Load model*/
    SAMPLE_SVP_TRACE_INFO("Cnn Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName,&s_stExtModel);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");
    /*CNN parameter initialization*/
    /*Cnn software parameters are set in SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit,
     if user has changed net struct, please make sure the parameter settings in
     SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit function are correct*/
    SAMPLE_SVP_TRACE_INFO("Cnn parameter initialization!\n");
    s_stExtNnieParam.pstModel = &s_stExtModel.stModel;
    s32Ret = SAMPLE_SVP_NNIE_Cnn_ParamInit(&stNnieCfg,&s_stExtNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Cnn_ParamInit failed!\n");
    SAMPLE_SVP_TRACE_INFO("NNIE AddTskBuf!\n");
    /*record tskBuf*/
    s32Ret = HI_MPI_SVP_NNIE_AddTskBuf(&(s_stExtNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_AddTskBuf failed!\n");
    SAMPLE_SVP_TRACE_INFO("NNIE AddTskBuf end!\n");
    return;

CNN_FAIL_1:
    /*Remove TskBuf*/
    s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(s_stExtNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_RemoveTskBuf failed!\n");

CNN_FAIL_0:
    SAMPLE_SVP_TRACE_INFO("Why \n");
    SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stExtNnieParam,&s_stExtModel);
    SAMPLE_COMM_SVP_CheckSysExit();
}


void NNIE_FACE_NNIE_EXTRACTOR_GET(char *pcSrcFile, float *feature_buff)
{
    stNnieCfg.pszPic= pcSrcFile;
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};

    /*Fill src data*/
    SAMPLE_SVP_TRACE_INFO("Face Extractor start!\n");
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg,&s_stExtNnieParam,&stInputDataIdx);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_1,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");
    SAMPLE_SVP_TRACE_INFO("Load Img!\n");
    /*NNIE process(process the 0-th segment)*/
    long spend;
    struct timespec start, next, end;
    clock_gettime(0, &start);
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stExtNnieParam,&stInputDataIdx,&stProcSegIdx,HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_1,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");
    SAMPLE_SVP_TRACE_INFO("Forward!\n");
    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    printf("\n[inference]===== TIME SPEND: %ld ms =====\n", spend);
    /*Software process*/
    /*Print results*/
    {
        printf("features:\n{\n");
        printf("stride: %d\n",s_stExtNnieParam.astSegData[0].astDst[0].u32Stride);
        printf("blob type :%d\n",s_stExtNnieParam.astSegData[0].astDst[0].enType);
        printf("{\n\tc :%d", s_stExtNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Chn);
        printf("\n\th :%d", s_stExtNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Height);
        printf("\n\tw :%d \n}\n", s_stExtNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Width);
        HI_S32* ps32Score = (HI_S32* )((HI_U8* )s_stExtNnieParam.astSegData[0].astDst[0].u64VirAddr);
        printf("blobs fc1:\n[");
        for(HI_U32 i = 0; i < 512; i++)
        {
            feature_buff[i] = *(ps32Score + i) / 4096.f;
        }
        
        printf("]\n}\n");
    }
    //s32Ret = SAMPLE_SVP_NNIE_PrintReportResult(&s_stExtNnieParam);
    //SVP_NNIE_MNET(&s_stExtNnieParam);
    clock_gettime(0, &next);
    spend = (next.tv_sec - end.tv_sec) * 1000 + (next.tv_nsec - end.tv_nsec) / 1000000;
    printf("\n[post process]===== TIME SPEND: %ld ms =====\n", spend);
    /*Print result*/
    // s32Ret = SAMPLE_SVP_NNIE_PrintReportResult(&s_stExtNnieParam);
    // SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,SAMPLE_SVP_NNIE_PrintReportResult failed!");
    return;
CNN_FAIL_1:
    /*Remove TskBuf*/
    SAMPLE_SVP_TRACE_INFO("Why1 \n");
    s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(s_stExtNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_RemoveTskBuf failed!\n");

CNN_FAIL_0:
    SAMPLE_SVP_TRACE_INFO("Why2 \n");
    SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stExtNnieParam,&s_stExtModel);
    SAMPLE_COMM_SVP_CheckSysExit();
}



/******************************************************************************
* function : Cnn sample signal handle
******************************************************************************/
void NNIE_FACE_EXTRACTOR_RELEASE(void)
{
    SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stExtNnieParam, &s_stExtModel);
    memset(&s_stExtNnieParam,0,sizeof(SAMPLE_SVP_NNIE_PARAM_S));
    memset(&s_stExtModel,0,sizeof(SAMPLE_SVP_NNIE_MODEL_S));
    SAMPLE_COMM_SVP_CheckSysExit();
}

void NNIE_FACE_PFPLD_INIT(char *pcModelName)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /*Set configuration parameter*/
    HI_U32 u32PicNum = 1;
    /*Set configuration parameter*/
    stNnieCfg.u32MaxInputNum = u32PicNum; //max input image num in each batch
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0;//set NNIE core

    /*Sys init*/
    SAMPLE_COMM_SVP_CheckSysInit();

    /*CNN Load model*/
    SAMPLE_SVP_TRACE_INFO("Cnn Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName,&s_stPfpldModel);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");
    /*CNN parameter initialization*/
    /*Cnn software parameters are set in SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit,
     if user has changed net struct, please make sure the parameter settings in
     SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit function are correct*/
    SAMPLE_SVP_TRACE_INFO("Cnn parameter initialization!\n");
    s_stPfpldNnieParam.pstModel = &s_stPfpldModel.stModel;
    s32Ret = SAMPLE_SVP_NNIE_Cnn_ParamInit(&stNnieCfg,&s_stPfpldNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Cnn_ParamInit failed!\n");
    SAMPLE_SVP_TRACE_INFO("NNIE AddTskBuf!\n");
    /*record tskBuf*/
    s32Ret = HI_MPI_SVP_NNIE_AddTskBuf(&(s_stPfpldNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_AddTskBuf failed!\n");
    SAMPLE_SVP_TRACE_INFO("NNIE AddTskBuf end!\n");
    return;

CNN_FAIL_1:
    /*Remove TskBuf*/
    s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(s_stPfpldNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_RemoveTskBuf failed!\n");

CNN_FAIL_0:
    SAMPLE_SVP_TRACE_INFO("Why \n");
    SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stPfpldNnieParam,&s_stPfpldModel);
    SAMPLE_COMM_SVP_CheckSysExit();
}


void NNIE_FACE_PFPLD_GET(char *pcSrcFile, float *landmarks_buff, float *angles_buff)
{
    stNnieCfg.pszPic= pcSrcFile;
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};

    /*Fill src data*/
    SAMPLE_SVP_TRACE_INFO("Face Extractor start!\n");
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg,&s_stPfpldNnieParam,&stInputDataIdx);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_1,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");
    SAMPLE_SVP_TRACE_INFO("Load Img!\n");
    /*NNIE process(process the 0-th segment)*/
    long spend;
    struct timespec start, next, end;
    clock_gettime(0, &start);
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stPfpldNnieParam,&stInputDataIdx,&stProcSegIdx,HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_1,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");
    SAMPLE_SVP_TRACE_INFO("Forward!\n");
    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    printf("\n[inference]===== TIME SPEND: %ld ms =====\n", spend);
    /*Software process*/
    /*Print results*/
    {
        printf("features:\n{\n");
        printf("stride: %d\n",s_stPfpldNnieParam.astSegData[0].astDst[0].u32Stride);
        printf("blob type :%d\n",s_stPfpldNnieParam.astSegData[0].astDst[0].enType);
        printf("{\n\tc :%d", s_stPfpldNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Chn);
        printf("\n\th :%d", s_stPfpldNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Height);
        printf("\n\tw :%d \n}\n", s_stPfpldNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Width);
        HI_S32* ps32Score = (HI_S32* )((HI_U8* )s_stPfpldNnieParam.astSegData[0].astDst[0].u64VirAddr);
        for(HI_U32 i = 0; i < 196; i++)
        {
            landmarks_buff[i] = *(ps32Score + i) / QUANT_BASE;
        }
        ps32Score = (HI_S32* )((HI_U8* )s_stPfpldNnieParam.astSegData[0].astDst[1].u64VirAddr);
        for(HI_U32 i = 0; i < 3; i++)
            angles_buff[i] = *(ps32Score + i)*180.f / QUANT_BASE / SAMPLE_PI;
    }
    clock_gettime(0, &next);
    spend = (next.tv_sec - end.tv_sec) * 1000 + (next.tv_nsec - end.tv_nsec) / 1000000;
    printf("\n[post process]===== TIME SPEND: %ld ms =====\n", spend);
    /*Print result*/
    // s32Ret = SAMPLE_SVP_NNIE_PrintReportResult(&s_stPfpldNnieParam);
    // SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,SAMPLE_SVP_NNIE_PrintReportResult failed!");
    return;
CNN_FAIL_1:
    /*Remove TskBuf*/
    SAMPLE_SVP_TRACE_INFO("Why1 \n");
    s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(s_stPfpldNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_RemoveTskBuf failed!\n");

CNN_FAIL_0:
    SAMPLE_SVP_TRACE_INFO("Why2 \n");
    SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stPfpldNnieParam,&s_stPfpldModel);
    SAMPLE_COMM_SVP_CheckSysExit();
}



/******************************************************************************
* function : Cnn sample signal handle
******************************************************************************/
void NNIE_FACE_PFPLD_RELEASE(void)
{
    SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stPfpldNnieParam, &s_stPfpldModel);
    memset(&s_stPfpldNnieParam,0,sizeof(SAMPLE_SVP_NNIE_PARAM_S));
    memset(&s_stPfpldModel,0,sizeof(SAMPLE_SVP_NNIE_MODEL_S));
    SAMPLE_COMM_SVP_CheckSysExit();
}
