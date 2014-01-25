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
static cv::BRISK briskOp;
static cv::ORB orbOp;
static cv::FREAK freakOp;
static cv::SURF surfOp;
static cv::SIFT siftOp;

static const int XSz = 320;
static const int YSz = 240;

/*static string depthString(const cv::Mat &m) {
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
}*/

/*static void showMatInfo(const cv::Mat &m) {
  std::cout 
    << "dim: " << m.dims
    << " depth:" << depthString(m) 
    << " r:" << m.rows 
    << " c:" << m.cols << "\n";
}*/

static cv::BFMatcher bitMatcher(cv::NORM_HAMMING);

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

struct FeatureInfo {
  FeatureInfo(int _rows, int _cols, int _type) : rows(_rows), cols(_cols), type(_type) { }
  int rows;
  int cols;
  int type;

  cv::Mat makeRandom(cv::RNG &rng) {
    cv::Mat img = cv::Mat::zeros(rows, cols, type);
    for(int y=0; y<cols; y++) {
      for(int x=0; x<rows; x++) {
        img.at<unsigned char>(x,y) = rng();
      }
    }
    return img;
  }
};

static FeatureInfo freakInfo(108,64,CV_8U);
static FeatureInfo briskInfo(391,64,CV_8U);
static FeatureInfo orbInfo(54,32,CV_8U);

static vector<cv::Mat> freakVocab;
static vector<cv::Mat> briskVocab;
static vector<cv::Mat> orbVocab;

const int VOCAB_SIZE = 64;

static void initVocabularies() {
  cv::RNG rng(0xdeadbeef);

  freakVocab.reserve(VOCAB_SIZE);
  briskVocab.reserve(VOCAB_SIZE);
  orbVocab.reserve(VOCAB_SIZE);
  for(int i=0; i<VOCAB_SIZE; i++) {
    cv::Mat freak = freakInfo.makeRandom(rng);
    freakVocab.push_back(freak);

    cv::Mat brisk = briskInfo.makeRandom(rng);
    briskVocab.push_back(brisk);

    cv::Mat orb = orbInfo.makeRandom(rng);
    orbVocab.push_back(orb);
  }
  std::cout << freakVocab[0].row(0) << "\n";
  std::cout << freakVocab[1].row(0) << "\n";
}

static vector<int> closestMatchTerms(cv::Mat desc, vector<cv::Mat> vocab) {
  vector<int> bestMatch(desc.cols, -1);
  vector<float> distance(desc.cols, 0.0f);

  for(int i=0; i<(int)vocab.size(); i++) {
    vector<cv::DMatch> matches;
    //std::cout << "desc "; showMatInfo(desc);
    //std::cout << "vocab[i] "; showMatInfo(vocab[i]);
    bitMatcher.match(desc, vocab[i], matches);

    // for each matching orb descriptor, keep this vocab word if it is closest to that descriptor
    for(int col=0; col<desc.cols; col++) {
      if(bestMatch[col] == -1 || distance[col] < matches[col].distance) {
        bestMatch[col] = i;
        distance[col] = matches[col].distance;
      }
    }
  }
  // return cols visual words as integers on [0,VOCAB_SIZE)
  return bestMatch;
}

static void init() {
  initKeypoints();
  initVocabularies();
}

static cv::Mat loadImage(string path) {
  cv::Mat fullData = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
  if(fullData.empty()) {
    return fullData;
  }
  
  // return smaller image
  cv::Mat imgData;
  cv::resize(fullData, imgData, cv::Size(XSz,YSz));
  return imgData;
}

struct ImgDesc {
  cv::Mat brisk;
  cv::Mat orb;
  cv::Mat freak;
  cv::Mat surf;
  cv::Mat sift;

  vector<float> compute(cv::Mat img) {
    vector<cv::KeyPoint> tmpkp;
    vector<int> terms;
    
    //tmpkp = vector<cv::KeyPoint>(keypoints);
    //surfOp.compute(img, tmpkp, surf);
    //
    vector<int> featureVector(VOCAB_SIZE*3, 0.0f);
    int featureSum = 0;

    tmpkp = vector<cv::KeyPoint>(keypoints);
    orbOp.compute(img, tmpkp, orb);
    terms = closestMatchTerms(orb, orbVocab);
    for(int idx : terms) {
      ++featureVector[idx];
      ++featureSum;
    }

    tmpkp = vector<cv::KeyPoint>(keypoints);
    freakOp.compute(img, tmpkp, freak);
    terms = closestMatchTerms(freak, freakVocab);
    for(int idx : terms) {
      ++featureVector[idx+VOCAB_SIZE];
      ++featureSum;
    }

    tmpkp = vector<cv::KeyPoint>(keypoints);
    briskOp.compute(img, tmpkp, brisk);
    terms = closestMatchTerms(brisk, briskVocab);
    for(int idx : terms) {
      ++featureVector[idx+2*VOCAB_SIZE];
      ++featureSum;
    }

    //tmpkp = vector<cv::KeyPoint>(keypoints);
    //siftOp.compute(img, tmpkp, sift);
    
    vector<float> normFeatures(featureVector.size(), 0.0f);
    for(size_t i=0; i<featureVector.size(); i++) {
      normFeatures[i] = float(featureVector[i])/float(featureSum);
    }

    return normFeatures;
  }
};

static void processImage(string path) {
  cv::Mat imgData = loadImage(path);

  if(imgData.empty()) {
    std::cerr << "Image " << path << " is empty, skipping.\n";
  }

  ImgDesc desc;
  vector<float> scores = desc.compute(imgData);

  printf("0 qid:0 ");
  int featureId = 1;
  for(float x : scores) {
    if(x > 0) {
      printf("%d:%1.5f ", featureId, x);
    }
    featureId++;
  }
  printf("# %s\n", path.c_str());

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

