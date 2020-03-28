#ifndef FD_TOOLS
#define FD_TOOLS
#include "anchor_generator.h"

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes);
void plot_pose_cube(cv::Mat& img, float yaw, float pitch, float roll, float tdx, float tdy, float size);
#endif // FD_TOOLS
