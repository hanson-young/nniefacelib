#ifndef __NNIE_FACE_H__
#define __NNIE_FACE_H__


#ifdef __cplusplus
#if __cplusplus
extern "C"{
#endif
#endif /* __cplusplus */

/******************************************************************************
* function : Retinaface Detector func
******************************************************************************/

void NNIE_FACE_DETECTOR_INIT(char *pcModelName, float threshold, int isLog);
void NNIE_FACE_DETECTOR_GET(char *pcSrcFile);
void NNIE_FACE_DETECTOR_RELEASE(void);

/******************************************************************************
* function : Face Recognition func
******************************************************************************/
void NNIE_FACE_EXTRACTOR_INIT(char *pcModelName);
void NNIE_FACE_NNIE_EXTRACTOR_GET(char *pcSrcFile, float *feature_buff);
void NNIE_FACE_EXTRACTOR_RELEASE(void);

/******************************************************************************
* function : Face Pose func
******************************************************************************/
void NNIE_FACE_PFPLD_INIT(char *pcModelName);
void NNIE_FACE_PFPLD_GET(char *pcSrcFile, float *landmarks_buff, float *angles_buff);
void NNIE_FACE_PFPLD_RELEASE(void);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */


#endif /* __NNIE_FACE_H__ */
