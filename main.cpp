#include <iostream>
#include <string>
#include <vector>
using std::string;
using std::vector;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

int main(int argc, char **argv) {
  vector<string> args;
  for(int i=0; i<argc; i++) {
    args.push_back(string(argv[i]));
  }

  for(const string& arg : args) {
    std::cout << "arg: " << arg << "\n";
  }
  
  cv::Mat imgData = cv::imread("buddy2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  std::cout << "Done loading!\n";

  cv::SurfFeatureDetector detector;
  cv::SurfDescriptorExtractor extractor;
  vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;

  detector.detect(imgData, keypoints);
  std::cout << "Done detection!\n";
  detector.compute(imgData, keypoints, descriptors);
  std::cout << "Done computation!\n";

  std::cout << "Done!\n";

}
