#include<iostream>
using namespace std;
#include<string>

#include "eigen3/Eigen/Dense"
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>

#include"../include/depth_segmentation.h"



int main( int argc, char** argv )
{
    //initialize log
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = 0;


    cv::Mat rgb, rgb1, rgb_image_;
    rgb_image_ = cv::imread("../date/rgb.png");


    cv::Mat dep, dep1, dep2, depth_image_;
    dep = cv::imread("../date/depth.pgm", -1);


    depth_image_ = cv::Mat(dep.size(), CV_16UC1);
    dep.convertTo(depth_image_, CV_16UC1, 1, 0);


    cv::Mat camera_intr_ = (cv::Mat_<double>(3, 3) << 500, 0, 325, 0, 519, 253, 0, 0, 1);
    cv::Mat label_map_;
    cv::Mat normal_map_;
    std::vector<cv::Mat> segment_masks_;

    // template<depth_segmentation::Segment Segment_>;
    std::vector<depth_segmentation::Segment> segments_;

    cv::Mat rgb_image;
    depth_segmentation::Params params_;

    clock_t startTime, endTime;
    startTime = clock();
    depth_segmentation::segmentSingleFrame(rgb_image_, depth_image_, camera_intr_, params_, 
                            &label_map_, &normal_map_, &segment_masks_, &segments_);
    endTime = clock();
    cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

    cv::waitKey( 0 );

    return 0;
}