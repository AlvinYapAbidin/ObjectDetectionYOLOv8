#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
using namespace cv::dnn;

// Look at postprocessing
// Print info about outs for any meaningful data
// blobFromImage

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;
 
// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;
 
// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);

std::vector<std::string> readLabels(std::string& labelFilepath) {
  std::vector<std::string> labels;
  std::string line;
  std::ifstream fp(labelFilepath);
  while (std::getline(fp, line)) {
    labels.push_back(line);
  }
  return labels;
}

std::vector<float> sigmoid(const std::vector<float>& m1) {
  const unsigned long vectorSize = m1.size();
  std::vector<float> output(vectorSize);
  for (unsigned i = 0; i != vectorSize; ++i) {
    output[i] = 1 / (1 + exp(-m1[i]));
  }
  return output;
}

void yoloPostProcessing(
    std::vector<Mat>& outs,
    std::vector<int>& keep_classIds,
    std::vector<float>& keep_confidences,
    std::vector<Rect2d>& keep_boxes,
    float conf_threshold,
    float iou_threshold
)
{

    // Retrieve
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect2d> boxes;

    cv::transposeND(outs[0], {0, 2, 1}, outs[0]);

    for (auto preds : outs){

        preds = preds.reshape(1, preds.size[1]); // [1, 8400, 85] -> [8400, 85]
        for (int i = 0; i < preds.rows; ++i)
        {
            // filter out non object
            float obj_conf = 1.0f;
            if (obj_conf < conf_threshold)
                continue;

            Mat scores = preds.row(i).colRange(4, preds.cols);
            double conf;
            Point maxLoc;
            minMaxLoc(scores, 0, &conf, 0, &maxLoc);

            conf = conf;
            if (conf < conf_threshold)
                continue;

            // get bbox coords
            float* det = preds.ptr<float>(i);
            double cx = det[0];
            double cy = det[1];
            double w = det[2];
            double h = det[3];

            // std::cout << "cx: " << cx << " cy: " << cy << " w: " << w << " h: " << h << " conf: " << conf << " idx: " << maxLoc.x << std::endl;
            // [x1, y1, x2, y2]
            
            boxes.push_back(Rect2d(cx - 0.5 * w, cy - 0.5 * h,
                                    cx + 0.5 * w, cy + 0.5 * h));
            
            classIds.push_back(maxLoc.x);
            confidences.push_back(conf);
        }
    }

    // NMS
    std::vector<int> keep_idx;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, keep_idx);

    for (auto i : keep_idx)
    {
        keep_classIds.push_back(classIds[i]);
        keep_confidences.push_back(confidences[i]);
        keep_boxes.push_back(boxes[i]);
    }
}

int main(int argc, char* argv[]) 
{
  int inpWidth = 128;
  int inpHeight = 128;
  std::string modelFilepath{
      "runs/detect/train4/weights/last.onnx"};
//   std::string labelFilepath{
    //   };
  //std::string imageFilepath{argv[1]};
  std::cout << "load image" << std::endl;
//   std::vector<std::string> labels{readLabels(labelFilepath)};
  cv::Mat image = cv::imread("./ball.png", cv::ImreadModes::IMREAD_COLOR);
  std::cout << "check 1" << std::endl;

  cv::Mat blob;
  cv::Scalar mean{0.4151, 0.3771, 0.4568};
  cv::Scalar std{0.2011, 0.2108, 0.1896};
  bool swapRB = false;
  bool crop = false;
  cv::dnn::blobFromImage(image, blob, 1.0, cv::Size(inpWidth, inpHeight), mean,
                         swapRB, crop);
  if (std.val[0] != 0.0 && std.val[1] != 0.0 && std.val[2] != 0.0) {
    cv::divide(blob, std, blob);
  }
    std::cout << "check 2" << std::endl;


  std::cout << "readnet1" << std::endl;
  cv::dnn::Net net = cv::dnn::readNet(modelFilepath);
  std::cout << "readnet2" << std::endl;
  net.setInput(blob);
  std::cout << "readnet3" << std::endl;
  std::vector<cv::Mat> outs;

  net.forward(outs, net.getUnconnectedOutLayersNames());
  std::cout << outs.size() << std::endl;
  
  std::vector<int> keep_classIds;
  std::vector<float> keep_confidences;
  std::vector<Rect2d> keep_boxes;
  float conf_threshold = 0.5;
  float iou_threshold = 0.05;

  yoloPostProcessing(
    outs,
    keep_classIds,
    keep_confidences,
    keep_boxes,
    conf_threshold,
    iou_threshold
  );

  std::cout << "size: " << keep_confidences.size() << " boxes size: " << keep_boxes.size() << std::endl;
  for (int i=0; i < keep_boxes.size(); i++)
  {
    
    std::cout << "x: " << keep_boxes[i].x << " y: " << keep_boxes[i].y << " width: " << keep_boxes[i].width << " height: " << keep_boxes[i].height << std::endl;
    printf("confidences: %f\n", keep_confidences[i]);
  }

  std::vector<Rect> boxes;


  for (auto box : keep_boxes)
  {
    // boxes.push_back(Rect(cvFloor(box.x), cvFloor(box.y), cvFloor(box.width - box.x), cvFloor(box.height - box.y)));
    boxes.push_back(Rect(cvFloor(box.x/128*image.rows), cvFloor(box.y/128*image.cols), cvFloor((box.width - box.x)/128*image.rows), cvFloor((box.height - box.y)/128*image.cols)));

  }

    // paramNet.blobRectsToImageRects(boxes, boxes, img.size());
  for (size_t idx = 0; idx < boxes.size(); ++idx)
  {
    Rect box = boxes[idx];
    //drawPrediction(keep_classIds[idx], keep_confidences[idx], box.x, box.y,
    // box.width + box.x, box.height + box.y, image);
    cv::rectangle(image, cv::Point(box.x, box.y), cv::Point(box.width + box.x, box.height + box.y), cv::Scalar(0,255, 255));
  }
  const std::string kWinName = "Yolo Object Detector";
  namedWindow(kWinName, WINDOW_NORMAL);
  imshow(kWinName, image);
  waitKey();
  destroyAllWindows();

//   cv::Mat prob = net.forward();
//   std::cout << "readnet4" << std::endl;
//   std::cout << prob << std::endl;
//   std::cout << "readnet5" << std::endl;

//   // Apply sigmoid
//   cv::Mat probReshaped = prob.reshape(1, prob.total() * prob.channels());
//   std::vector<float> probVec =
//       probReshaped.isContinuous() ? probReshaped : probReshaped.clone();
//   std::vector<float> probNormalized = sigmoid(probVec);

//   cv::Point classIdPoint;
//   double confidence;
//   minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
//   int classId = classIdPoint.x;
//   std::cout << " ID " << classId << " - " << labels[classId] << " confidence "
//             << confidence << std::endl;
}
