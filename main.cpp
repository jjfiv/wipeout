#include <iostream>
#include <string>
#include <cstdio>
#include <vector>
#include <fstream>
using std::string;
using std::vector;

#include <opencv2/core/core.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

typedef unsigned char u8;

static_assert(sizeof(u8) == sizeof(uchar), "u8=uchar");
static_assert(sizeof(u8) == 1, "u8=1");

template <class T>
static T constexpr min(T a, T b) {
  return (a < b) ? a : b;
}
static float constexpr max(float a, float b) {
  return (a > b) ? a : b;
}

static vector<cv::KeyPoint> keypoints;
static cv::BRISK briskOp;
static cv::ORB orbOp(500, 1.2f, 8, 16);
static cv::FREAK freakOp;
static cv::SURF surfOp;
static cv::SIFT siftOp;

static const int XSz = 320;
static const int YSz = 240;

static int _bitcount_cache[256] = {0};
static int slow_bitcount(u8 n) {
  int count = 0;
  while(n) {
    count += n & 1u;
    n >>= 1;
  }
  return count;
}
static void initBitcount() {
  for(int i=0; i<256; i++) {
    _bitcount_cache[i] = slow_bitcount(i);
  }
}

static int fast_bitcount(u8 n) {
  return _bitcount_cache[n];
}


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
  std::cerr 
    << "dim: " << m.dims
    << " depth:" << depthString(m) 
    << " r:" << m.rows 
    << " c:" << m.cols << "\n";
}*/

static cv::BFMatcher bitMatcher(cv::NORM_HAMMING);

static void initKeypoints() {
  //const int XN = 32*2;
  //const int YN = 24*2;
  //const float diameter = max(2,XSz / XN);
  //for(int x = 0; x < XN; x++) {
  //  for(int y = 0; y < YN; y++) {
  //    float xp = float(XSz) / float(x);
  //    float yp = float(YSz) / float(y);

  //    keypoints.push_back(cv::KeyPoint(xp, yp, diameter));
  //  }
  //}

  cv::RNG rng(0xdeadbeef);
  for(int n=0; n<2000; n++) {
    float x = rng.uniform(0,XSz);
    float y = rng.uniform(0,YSz);
    keypoints.push_back(cv::KeyPoint(x, y, rng.uniform(1.0f, 5.0f)));
  }

  std::cerr << "init " << keypoints.size() << " keypoints!\n";
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

static FeatureInfo freakInfo(1376,64,CV_8U);
static FeatureInfo briskInfo(1631,64,CV_8U);
static FeatureInfo orbInfo(1559,32,CV_8U);

static vector<cv::Mat> freakVocab;
static vector<cv::Mat> briskVocab;
static vector<cv::Mat> orbVocab;

const int VOCAB_SIZE = 1024;

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
  //std::cerr << freakVocab[0].row(0) << "\n";
  //std::cerr << freakVocab[1].row(0) << "\n";
}

static vector<int> closestMatchTerms(cv::Mat desc, vector<cv::Mat> vocab) {
  int cols = desc.cols;
  int rows = desc.rows;

  assert(desc.cols == vocab[0].cols);
  if(desc.rows != vocab[0].rows) {
    std::cerr << "desc.rows: " << desc.rows << "\n";
    std::cerr << "vocab[0].rows: " << vocab[0].rows << "\n";
  }
  assert(desc.rows == vocab[0].rows);

  vector<int> bestMatch(rows, -1);
  vector<int> distance(rows, 0);

  for(int i=0; i<(int)vocab.size(); i++) {
    for(int x=0; x<rows; x++) {
      int thisRowDistance = 0;
      for(int y=0; y<cols; y++) {
        u8 ai = desc.at<uchar>(x,y);
        u8 bi = vocab[i].at<uchar>(x,y);

        // bit distance is xor of ai, bi
        thisRowDistance += fast_bitcount(ai ^ bi);
      }

      // keep this word if it's the best match so far
      if(bestMatch[x] == -1 || thisRowDistance < distance[x]) {
        bestMatch[x] = i;
        distance[x] = thisRowDistance;
      }
    }
  }
  // return cols visual words as integers on [0,VOCAB_SIZE)
  return bestMatch;
}

static void init() {
  initBitcount();
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

vector<float> computeBOW(cv::Mat img) {
  cv::Mat brisk;
  cv::Mat orb;
  cv::Mat freak;
  cv::Mat surf;
  cv::Mat sift;

  vector<cv::KeyPoint> tmpkp;
  vector<int> terms;

  //tmpkp = vector<cv::KeyPoint>(keypoints);
  //surfOp.compute(img, tmpkp, surf);
  //
  vector<int> featureVector(VOCAB_SIZE*3, 0.0f);
  int featureSum = 0;

  //tmpkp = vector<cv::KeyPoint>(keypoints);
  //orbOp.compute(img, tmpkp, orb);
  //terms = closestMatchTerms(orb, orbVocab);
  //for(int idx : terms) {
  //  ++featureVector[idx];
  //  ++featureSum;
  //}

  //tmpkp = vector<cv::KeyPoint>(keypoints);
  //freakOp.compute(img, tmpkp, freak);
  //terms = closestMatchTerms(freak, freakVocab);
  //for(int idx : terms) {
  //  ++featureVector[idx+VOCAB_SIZE];
  //  ++featureSum;
  //}

  tmpkp = vector<cv::KeyPoint>(keypoints);
  briskOp.compute(img, tmpkp, brisk);
  terms = closestMatchTerms(brisk, briskVocab);
  for(int idx : terms) {
    ++featureVector[idx+2*VOCAB_SIZE];
    ++featureSum;
  }

  //tmpkp = vector<cv::KeyPoint>(keypoints);
  //siftOp.compute(img, tmpkp, sift);

  int max = 0;
  for(int ft : featureVector) {
    if(ft > max) {
      max = ft;
    }
  }
  //std::cerr << "featureSum: " << featureSum << "\n";
  vector<float> normFeatures(featureVector.size(), 0.0f);
  for(size_t i=0; i<featureVector.size(); i++) {
    normFeatures[i] = 10.0f *(float(featureVector[i]) / float(max)); //float(featureSum);
  }

  return normFeatures;
}

static vector<float> processImage(string path) {
  cv::Mat imgData = loadImage(path);

  if(imgData.empty()) {
    std::cerr << "Image `" << path << "' is empty, skipping.\n";
    return vector<float>();
  }

  vector<float> scores = computeBOW(imgData);
  return scores;
}

static vector<string> slurpLines(const string &path) {
  vector<string> lines;
  std::ifstream fp;
  fp.open(path);

  while(fp) {
    string line;
    std::getline(fp, line);
    if(line.size()) {
      lines.push_back(line);
    }
  }

  return lines;
}

template <class T>
static vector<T> subvec(const vector<T> &in, int start, int end) {
  vector<T> result;
  for(int i=start; i<end && i < in.size(); i++) {
    result.push_back(in[i]);
  }
  return result;
}

template <class T>
static vector<T> concat(const vector<T> &a, const vector<T> &b) {
  vector<T> conc(a);
  conc.reserve(a.size()+b.size());
  for(const T& ai : a) {
    conc.push_back(ai);
  }
  for(const T& bi : b) {
    conc.push_back(bi);
  }
  return conc;
}

static void trainSVM(vector<vector<float>> pos, vector<vector<float>> neg, const char * modelPath) {
  // default params taken from:
  // http://docs.opencv.org/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html
  CvSVMParams params;
  params.svm_type = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 4000, 1e-6);
  params.C = 30;


  const int dataPoints = pos.size() + neg.size();
  const int dataWidth = pos.at(0).size();
  const int numPos = pos.size();
  const int numNeg = neg.size();

  // set up labels; positive first
  vector<float> labels(dataPoints, 0.0);
  for(int i=0; i<numPos; i++) {
    labels[i] = (float) i;
  }

  cv::Mat training = cv::Mat::zeros(dataPoints, dataWidth, CV_32FC1);
  for(int i=0; i<numPos; i++) {
    for(int x=0; x<dataWidth; x++) {
      training.at<float>(i,x) = pos[i][x];
    }
  }
  for(int i=0; i<numNeg; i++) {
    for(int x=0; x<dataWidth; x++) {
      training.at<float>(i+numPos,x) = neg[i][x];
    }
  }

  std::cerr << "Start train!\n";
  // create and train
  CvSVM svm;
  svm.train(training, cv::Mat(labels), cv::Mat(), cv::Mat(), params);

  //std::cerr << cv::Mat(labels) << "\n";

  int posD = 0;
  for(auto score : pos) {
    posD += (int) (svm.predict(cv::Mat(score)) > 0);
  }
  std::cerr << "predict on train-pos: " << posD << "/" << pos.size() << "\n";
  std::cerr << "End train!\n";
  
  svm.save(modelPath);
}

static void showPath(const char *title, string path) {
  std::cout << "<img src=\"" << path << "\" /><br />\n";
}

struct Image {
  Image(string p) : path(p) { }
  string path;
  vector<float> scores;
};

int main(int argc, char **argv) {
  init();

  if(argc != 3) {
    std::cerr << "Expected positive.list negatives.list\n";
    return -1;
  }

  vector<string> pos = slurpLines(argv[1]);
  vector<string> neg = slurpLines(argv[2]);

  std::cerr << pos.size() << " positives\n";
  std::cerr << neg.size() << " negatives\n";

  vector<vector<float>> posd;
  vector<vector<float>> negd;

  for(string img : pos) {
    posd.push_back(processImage(img));
  }
  for(string img : neg) {
    negd.push_back(processImage(img));
  }

  //for(int numPos = 2; numPos < (posd.size() - 1); numPos++) {
  {
    const int numPos = 15;
    const int numNeg = 75;
    vector<vector<float>> trainPos = subvec(posd, 0, numPos);
    vector<vector<float>> heldOutPos = subvec(posd, numPos, posd.size());

    vector<vector<float>> trainNeg = subvec(negd, 0, numNeg);
    vector<vector<float>> heldOutNeg = subvec(negd, numNeg, negd.size());

    const char* const p2path = "p2.xml";
    trainSVM(trainPos, trainNeg, p2path);
    CvSVM p2;
    p2.load(p2path);

    vector<vector<float>> test = concat(heldOutPos, heldOutNeg);

    int truePos = 0;
    int falsePos = 0;
    int rank = 0;
    for(auto scores : test) {
      float response = p2.predict(cv::Mat(scores));
      if(rank < (heldOutPos.size())) {
        truePos += (response > 0.5);
        if(response > 0.5) {
          showPath("detected!", pos[rank]);
        }
      } else {
        falsePos += (response > 0.5);
        if(response > 0.5) {
          showPath("detected!", neg[rank-heldOutPos.size()]);
        }
      }
      rank++;
    }
    std::cout 
      << "truePos=" << truePos << "/" << heldOutPos.size() 
      << " falsePos=" << falsePos << "/" << heldOutNeg.size() << "\n";
  }
  


  

  /*
  vector<string> imgs;
  for(int i=1; i<argc; i++) {
    imgs.push_back(string(argv[i]));
  }

  for(const string& img : imgs) {
    processImage(img);
  }
  */
  
}

