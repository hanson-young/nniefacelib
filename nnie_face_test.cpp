#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include "nnie_face_api.h"

void NNIE_FACE_DET_TEST(void)
{
    printf("Face Detector Start!!!");
    char *pcModelName = "./data/nnie_model/face/mnet_640_inst.wk";
    char *pcSrcFile1 = "./data/nnie_image/rgb_planar/det41.bgr";
    char *pcSrcFile2 = "./data/nnie_image/rgb_planar/det39.bgr";
    float threshold = 0.7;
    int isLog = 1;
    NNIE_FACE_DETECTOR_INIT(pcModelName, threshold, isLog);

    for(int i = 0; i < 2; ++i)
    {
        printf("\n\n ========== =========== ========== ==========\n\n");

        if(i%2)
            NNIE_FACE_DETECTOR_GET(pcSrcFile1);
        else
            NNIE_FACE_DETECTOR_GET(pcSrcFile2);
    }

    NNIE_FACE_DETECTOR_RELEASE();
}


void NNIE_FACE_EXT_TEST(void)
{
    printf("Face Extractor Start!!!");
    char *pcModelName = "./data/nnie_model/face/mobilefacenet_inst.wk";
    char *pcSrcFile1 = "./data/nnie_image/rgb_planar/id10.bgr";
    char *pcSrcFile2 = "./data/nnie_image/rgb_planar/id11.bgr";
    float threshold = 0.7;
    int isLog = 0;
    NNIE_FACE_EXTRACTOR_INIT(pcModelName);
    float features[512] = {0};
    for(int i = 0; i < 2; ++i)
    {
        printf("\n\n ========== =========== ========== ==========\n\n");

        if(i%2)
            NNIE_FACE_NNIE_EXTRACTOR_GET(pcSrcFile1, features);
        else
            NNIE_FACE_NNIE_EXTRACTOR_GET(pcSrcFile2, features);
        printf("blobs fc1:\n[");
        
        for(int f = 0; f < 512; ++f)
        {
            printf("%f ,",features[f]);
            features[f] = 0;
        }
        printf("]\n");
    }

    NNIE_FACE_EXTRACTOR_RELEASE();
}

static char **s_ppChCmdArgv = NULL;

/******************************************************************************
* function : show usage
******************************************************************************/
void SAMPLE_SVP_Usage(char* pchPrgName)
{
    printf("Usage : %s <index> \n", pchPrgName);
    printf("index:\n");
    printf("\t 4) Cnn(Read File).\n");
}

/******************************************************************************
* function : nnie sample
******************************************************************************/

int main(int argc, char *argv[])

{
    int s32Ret = 1;

    if (argc < 2 || argc > 2)
    {
        SAMPLE_SVP_Usage(argv[0]);
        return 0;
    }

    if (!strncmp(argv[1], "-h", 2))
    {
        SAMPLE_SVP_Usage(argv[0]);
        return 1;
    }

    s_ppChCmdArgv = argv;

    switch (*argv[1])
    {
        case '0':
            {
                NNIE_FACE_DET_TEST();
            }
            break;
        case '1':
            {
                NNIE_FACE_EXT_TEST();
            }
            break;
        default :
            {
                SAMPLE_SVP_Usage(argv[0]);
            }
            break;
    }

    return s32Ret;
}



