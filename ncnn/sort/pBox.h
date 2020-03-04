#ifndef PBOX_H
#define PBOX_H
#include <stdlib.h>
#include <iostream>
#include "opencv2/core/types.hpp"
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;
#define mydataFmt float


struct pBox
{
	mydataFmt *pdata;
	int width;
	int height;
	int channel;
};

struct pRelu
{
    mydataFmt *pdata;
    int width;
};

struct Weight
{
	mydataFmt *pdata;
    mydataFmt *pbias;
    int lastChannel;
    int selfChannel;
	int kernelSize;
    int stride;
    int pad;
};

struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    mydataFmt ppoint[10];
    mydataFmt regreCoord[4];
};

struct BoundingBox {
    Rect_<float> rect;
    Point2f points[5];

    BoundingBox(const Bbox &box) {
        rect = Rect_<float>(box.x1, box.y1, box.x2, box.y2);
        for (int i = 0; i < 5; ++i) {
            points[i] = Point2f(box.ppoint[i], box.ppoint[i + 5]);
        }
    }

    template<class T> double dist(const T &a, const T &b) {
        return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
    }

    double relative(double a, double b) {
        return min(a, b) / max(a, b);
    }

    Point2f midpoint(const Point2f &a, const Point2f &b) {
        return Point2f((a.x + b.x) / 2, (a.y + b.y) / 2);
    }

    bool ccw(Point2f a, Point2f b, Point2f c) {
        b.x -= a.x; b.y -= a.y;
        c.x -= a.x; c.y -= a.y;
        return b.x * c.y < b.y * c.x; // oppose to common coordinates system in math, so weird openCV
    }

    bool is_frontal() {
        // 0 --- 1
        // -- 2 --
        // 3 --- 4
        if (ccw(points[0], points[1], points[2])) return false;
        if (ccw(points[1], points[4], points[2])) return false;
        if (ccw(points[4], points[3], points[2])) return false;
        if (ccw(points[3], points[0], points[2])) return false;

        double up = dist(points[2], midpoint(points[0], points[1]));
        double down = dist(points[2], midpoint(points[3], points[4]));
        double left = dist(points[2], midpoint(points[0], points[3]));
        double right = dist(points[2], midpoint(points[1], points[4]));
        //printf("up-down=%.2f left-right=%.2f\n", up / down, left / right); 
        return relative(up, down) > 0.5 && relative(left, right) > 0.3;

    }
};

struct orderScore
{
    mydataFmt score;
    int oriOrder;
};

void freepBox(struct pBox *pbox);
void freeWeight(struct Weight *weight);
void freepRelu(struct pRelu *prelu);
void pBoxShow(const struct pBox *pbox);
void pBoxShowE(const struct pBox *pbox,int channel, int row);
void weightShow(const struct Weight *weight);
void pReluShow(const struct pRelu *prelu);
#endif