#include "tools.h"


void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes) {
    filterOutBoxes.clear();
    if(boxes.size() == 0)
        return;
    std::vector<size_t> idx(boxes.size());

    for(unsigned i = 0; i < idx.size(); i++)
    {
        idx[i] = i;
    }

    //descending sort
    sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

    while(idx.size() > 0)
    {
        int good_idx = idx[0];
        filterOutBoxes.push_back(boxes[good_idx]);

        std::vector<size_t> tmp = idx;
        idx.clear();
        for(unsigned i = 1; i < tmp.size(); i++)
        {
            int tmp_i = tmp[i];
            float inter_x1 = std::max( boxes[good_idx][0], boxes[tmp_i][0] );
            float inter_y1 = std::max( boxes[good_idx][1], boxes[tmp_i][1] );
            float inter_x2 = std::min( boxes[good_idx][2], boxes[tmp_i][2] );
            float inter_y2 = std::min( boxes[good_idx][3], boxes[tmp_i][3] );

            float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
            float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

            float inter_area = w * h;
            float area_1 = (boxes[good_idx][2] - boxes[good_idx][0] + 1) * (boxes[good_idx][3] - boxes[good_idx][1] + 1);
            float area_2 = (boxes[tmp_i][2] - boxes[tmp_i][0] + 1) * (boxes[tmp_i][3] - boxes[tmp_i][1] + 1);
            float o = inter_area / (area_1 + area_2 - inter_area);           
            if( o <= threshold )
                idx.push_back(tmp_i);
        }
    }
}

void plot_pose_cube(cv::Mat& img, float yaw, float pitch, float roll, float tdx, float tdy, float size){
    float p = pitch * CV_PI / 180;
    float y = -(yaw * CV_PI / 180);
    float r = roll * CV_PI / 180;
    int face_x = tdx - 0.50 * size;
    int face_y = tdy - 0.50 * size;

    int x1 = size * (cos(y) * cos(r)) + face_x;
    int y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y;
    int x2 = size * (-cos(y) * sin(r)) + face_x;
    int y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y;
    int x3 = size * (sin(y)) + face_x;
    int y3 = size * (-cos(y) * sin(p)) + face_y;

    //Draw base in red
    cv::line(img, cv::Point(int(face_x), int(face_y)), cv::Point(int(x1), int(y1)), cv::Scalar(0, 0, 255), 3);
    cv::line(img, cv::Point(int(face_x), int(face_y)), cv::Point(int(x2), int(y2)), cv::Scalar(0, 0, 255), 3);
    cv::line(img, cv::Point(int(x2), int(y2)), cv::Point(int(x2 + x1 - face_x), int(y2 + y1 - face_y)), cv::Scalar(0, 0, 255), 3);
    cv::line(img, cv::Point(int(x1), int(y1)), cv::Point(int(x1 + x2 - face_x), int(y1 + y2 - face_y)), cv::Scalar(0, 0, 255), 3);
    //Draw pillars in blue
    cv::line(img, cv::Point(int(face_x), int(face_y)), cv::Point(int(x3), int(y3)), cv::Scalar(255, 0, 0), 2);
    cv::line(img, cv::Point(int(x1), int(y1)), cv::Point(int(x1 + x3 - face_x), int(y1 + y3 - face_y)), cv::Scalar(255, 0, 0), 2);
    cv::line(img, cv::Point(int(x2), int(y2)), cv::Point(int(x2 + x3 - face_x), int(y2 + y3 - face_y)), cv::Scalar(255, 0, 0), 2);
    cv::line(img, cv::Point(int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
                 cv::Point(int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), cv::Scalar(255, 0, 0), 2);
    //Draw top in green
    cv::line(img, cv::Point(int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
                 cv::Point(int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), cv::Scalar(0, 255, 0), 2);
    cv::line(img, cv::Point(int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
                 cv::Point(int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), cv::Scalar(0, 255, 0), 2);
    cv::line(img, cv::Point(int(x3), int(y3)), cv::Point(int(x3 + x1 - face_x), int(y3 + y1 - face_y)), cv::Scalar(0, 255, 0), 2);
    cv::line(img, cv::Point(int(x3), int(y3)), cv::Point(int(x3 + x2 - face_x), int(y3 + y2 - face_y)), cv::Scalar(0, 255, 0), 2);

}
    