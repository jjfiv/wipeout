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

static float constexpr max(float a, float b) {
  return (a > b) ? a : b;
}

static vector<cv::KeyPoint> keypoints;
static cv::BRISK brisk;
static cv::ORB orb;
static cv::FREAK freak;
static cv::SURF surf;
static cv::SIFT sift;

static cv::Mat briskVocab;
static cv::Mat orbVocab;
static cv::Mat freakVocab;
static cv::Mat surfVocab;
static cv::Mat siftVocab;

static const int XSz = 320;
static const int YSz = 240;

string depthString(const cv::Mat &m) {
  switch(m.depth()) {
    case CV_8U: return "CV_8U";
    case CV_8S: return "CV_8S";
    case CV_16U: return "CV_16U";
    case CV_16S: return "CV_16S";
    case CV_32S: return "CV_32S";
    case CV_32F: return "CV_32F";
    case CV_64F: return "CV_64F";
    default: break;
  }
  return "unk";
}

struct FeatureInfo {
  FeatureInfo(int _rows, int _cols, int _depth) : rows(_rows), cols(_cols), depth(_depth) { }
  int rows;
  int cols;
  int depth;
};

FeatureInfo freakInfo(108,64,8);
FeatureInfo briskInfo(391,64,8);
FeatureInfo orbInfo(54,64,8);

void showMatInfo(const cv::Mat &m) {
  std::cout 
    << "dim: " << m.dims
    << " depth:" << depthString(m) 
    << " r:" << m.rows 
    << " c:" << m.cols << "\n";
}

static void initKeypoints() {
  const int XN = 32*2;
  const int YN = 24*2;
  const float diameter = max(2,XSz / XN);
  for(int x = 0; x < XN; x++) {
    for(int y = 0; y < YN; y++) {
      float xp = float(XSz) / float(x);
      float yp = float(YSz) / float(y);

      keypoints.push_back(cv::KeyPoint(xp, yp, diameter));
    }
  }
}

static void init() {
  initKeypoints();
}

static void processImage(string path) {
  
  cv::Mat imgData;
  {
    cv::Mat fullData = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    if(fullData.empty()) {
      std::cerr << "Image " << path << " is empty, skipping.\n";
      return;
    }
    cv::resize(fullData, imgData, cv::Size(XSz,YSz));
  }

  if(keypoints.size() == 0) {
  }
  

  cv::Mat brisk_desc;
  cv::Mat orb_desc;
  cv::Mat freak_desc;
  cv::Mat surf_desc;
  cv::Mat sift_desc;
  
  vector<cv::KeyPoint> tmpkp;
  //tmpkp = vector<cv::KeyPoint>(keypoints);
  //surf.compute(imgData, tmpkp, surf_desc);
  
  tmpkp = vector<cv::KeyPoint>(keypoints);
  orb.compute(imgData, tmpkp, orb_desc);
  
  tmpkp = vector<cv::KeyPoint>(keypoints);
  freak.compute(imgData, tmpkp, freak_desc);
  
  tmpkp = vector<cv::KeyPoint>(keypoints);
  brisk.compute(imgData, tmpkp, brisk_desc);
  
  //tmpkp = vector<cv::KeyPoint>(keypoints);
  //sift.compute(imgData, tmpkp, sift_desc);


  std::cout << "surf "; showMatInfo(surf_desc);   
  std::cout << "sift "; showMatInfo(sift_desc);   
  std::cout << "orb "; showMatInfo(orb_desc);     
  std::cout << "freak "; showMatInfo(freak_desc); 
  std::cout << "brisk "; showMatInfo(brisk_desc); 
  //
  //std::cout << surf_desc.row(0);
  //std::cout << sift_desc.row(0);
  //std::cout << orb_desc.row(0);
  //std::cout << freak_desc.row(0);
  //std::cout << brisk_desc.row(0);
  
  /*printf("0 qid:0 ");
  int featureId = 1;
  for(auto it = surf_desc.begin<double>(); it != surf_desc.end<double>(); ++it) {
    double val = *it;
    if(int(val*100000) > 2) {
      printf("%d:%1.5f ", featureId, *it);
    }
    featureId++;
  }*/
  printf("# buddy\n");

  //std::cerr << "Wrote " << (featureId-1) << " features!\n";
}

int main(int argc, char **argv) {
  init();

  vector<string> imgs;
  for(int i=1; i<argc; i++) {
    imgs.push_back(string(argv[i]));
  }

  for(const string& img : imgs) {
    processImage(img);
  }
  
}

