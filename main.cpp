#include <iostream>
#include <string>
#include <cstdio>
#include <vector>
using std::string;
using std::vector;

#include <opencv2/core/core.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

static void processImage(string path) {
  const int XN = 64;
  const int YN = 48;
  const int XSz = 640;
  const int YSz = 480;
  
  cv::Mat imgData;
  {
    cv::Mat fullData = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    if(fullData.empty()) {
      std::cerr << "Image " << path << " is empty, skipping.\n";
      return;
    }
    cv::resize(fullData, imgData, cv::Size(XSz,YSz));
  }
  std::cerr << "Done loading & resizing!\n";

  vector<cv::KeyPoint> keypoints;

  const float diameter = XSz / XN;
  for(int x = 0; x < XN; x++) {
    for(int y = 0; y < YN; y++) {
      float xp = float(XSz) / float(x);
      float yp = float(YSz) / float(y);

      keypoints.push_back(cv::KeyPoint(xp, yp, diameter));
    }
  }

  //algorithm.detect(imgData, keypoints);
  std::cerr << keypoints.size() << " keypoints!\n";
  cv::SURF surf;
  cv::Mat surf_desc;
  surf.compute(imgData, keypoints, surf_desc);

  cv::FREAK freak;
  cv::Mat freak_desc;
  freak.compute(imgData, keypoints, freak_desc);
  
  std::cerr << "Done computation!\n";

  printf("0 qid:0 ");
  int featureId = 1;
  for(auto it = surf_desc.begin<double>(); it != surf_desc.end<double>(); ++it) {
    double val = *it;
    if(int(val*10000) != 0) {
      printf("%d:%1.7f ", featureId, *it);
    }
    featureId++;
  }
  printf("# buddy\n");

  std::cerr << "Wrote " << (featureId-1) << " features!\n";
}

int main(int argc, char **argv) {
  vector<string> imgs;
  for(int i=1; i<argc; i++) {
    imgs.push_back(string(argv[i]));
  }

  for(const string& img : imgs) {
    std::cerr << "process img: " << img << "\n";
    processImage(img);
  }
  
}

