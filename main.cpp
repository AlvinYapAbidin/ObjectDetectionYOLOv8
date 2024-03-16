// C++ modules
#include <fstream>
#include <vector>
#include <string>
#include <random>

// OpenCV modules
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/* Steps
  1. Load the ONNX model & video
  2. Run through each frame
  3. Preprocess the frame
  4. Run the model
  6. Post-process the output
  7. Display/Save the output
*/

struct Detection
{
  // This structure is intended to store the detection data
  int class_id{0};
  std::string className{};
  float confidence{0.0};
  cv::Scalar color{};
  cv::Rect box{};
};

void loadModel(std::string& modelPath, cv::dnn::Net& net)
{
  net = cv::dnn::readNetFromONNX(modelPath);
  std::cout << "\nRunning on CPU" << std::endl;
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void runInference(cv::Mat& frame, cv::dnn::Net& net, cv::Mat& blob, std::vector<cv::Mat>& outputs)
{
  cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
  net.setInput(blob);
  net.forward(outputs, net.getUnconnectedOutLayersNames());
}

void displayResults(cv::Mat& frame, std::vector<Detection>& detections)
{
  for (const auto& detection : detections) 
  {
      cv::rectangle(frame, detection.box, detection.color, 2);
      std::string label = detection.className + ": " + std::to_string(detection.confidence);
      int baseLine;
      cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
      cv::rectangle(frame, cv::Point(detection.box.x, detection.box.y - labelSize.height),
                    cv::Point(detection.box.x + labelSize.width, detection.box.y + baseLine), detection.color, cv::FILLED);
      cv::putText(frame, label, cv::Point(detection.box.x, detection.box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
  }

  cv::imshow("Frame", frame);
  if (cv::waitKey(50)); // 10 millisecond delay; Exit on ESC
}

Detection applyNMS(const std::vector<Detection>& detections, std::vector<cv::Rect>& bboxes, std::vector<float>& confidence,std::vector<int>& class_ids,std::vector<std::string>& classes, float& modelScoreThreshold, float& modelNMSThreshold)
{
  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(bboxes, confidence, modelScoreThreshold, modelNMSThreshold, nms_result);

    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
      std::cout;
      // Loop through each bounding box accepted by the NMS filter
      int index = nms_result[i];

      Detection result;
      result.class_id = class_ids[index];
      result.confidence = confidence[index];

      if (result.class_id == 0)
      {
        result.color = cv::Scalar(100, 100, 255);
      }
      else
      {
        result.color = cv::Scalar(255, 100, 100);
      }

      result.className = classes[result.class_id];
      result.box = bboxes[index];

      return result;
    }
}


int main(int argc, char**argv)
{

  /////////////////////      Declare Variables and Paths      /////////////////////

  std::string modelFilepath{"best.onnx"};
  std::string videoFilepath{"videos/shot1.mp4"};
  std::string classesFilePath{"classes.txt"};

  cv::VideoCapture cap(videoFilepath);
  
  if (!cap.isOpened())
  {
    return -1;
  }
  std::cout << "checkpoint" << std::endl;

  // Required variables
  int inpWidth  = 640;
  int inpHeight = 640;

  std::vector<std::string> classes{};
  std::vector<Detection> detections{};

  float modelConfidenseThreshold {0.25};
  float modelScoreThreshold      {0.50};
  float modelNMSThreshold        {0.50};

  cv::dnn::Net net;
  loadModel(modelFilepath, net);

  // Load class names from classesFilePath into classes vector
  std::ifstream ifs(classesFilePath.c_str());
  std::string line;
  while (getline(ifs, line)) classes.push_back(line);

  cv::Mat frame;
  while (true)
  {
    detections.clear(); // clears the detection at the start of the loop

    cap >> frame; // Go through video frame-by-frame. Each frame that is stored in 'cap' is stored in 'frame'
    if (frame.empty()) break;

    /////////////////////      Inference      /////////////////////
    
    cv::Mat blob;
    std::vector<cv::Mat> outputs;
    runInference(frame, net, blob, outputs);

    /////////////////////      Data Transmutation      /////////////////////

    /*  yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h] + confidence[c])
        Reshape and transpose to prepare for iteration over yolov8 model's output to process each detection.
        'row' indicates no. of items and 'dimensions' represents the combined data for each prediction */
    int dimensions = outputs[0].size[1];
    int rows = outputs[0].size[2];
    
    outputs[0] = outputs[0].reshape(1, dimensions);
    std::cout << "reshape: " << outputs[0].size << std::endl;
    cv::transpose(outputs[0], outputs[0]);
    std::cout << "transpose: " << outputs[0].size << std::endl;

    // Pointer initialization for model output data for indexing
    float *data = (float *)outputs[0].data;
    float *base_data = (float *)outputs[0].data;

    // Scaling factors which will be used to scale bounding box coordinates later from the model's output to the original input image
    cv::Mat modelInput = frame;
    float x_factor = modelInput.cols / inpWidth;
    float y_factor = modelInput.rows / inpHeight;

    /////////////////////     Post-processing Stage     /////////////////////
    std::vector<int> class_ids;
    std::vector<float> confidence;
    std::vector<cv::Rect> bboxes;

    for (int i = 0; i < rows; ++i)
    {
      // Skip the first 4 outputs which are the model's bbox coordinates (x, y, w, h) to get the new pointer of the scores
      float *classes_scores = data+4;

      cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
      cv::Point class_id;
      double maxClassScore;

      // Store the max score with the associated class id
      cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

      if (maxClassScore > modelScoreThreshold)
      {
        // Store the confidence, class ids and bounding boxes into vectors
        confidence.push_back(maxClassScore);
        class_ids.push_back(class_id.x); // x component from cv::Point structure returned in maxLoc has the max score

        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];
 
        // Bounding box calculations
        int left  = int((x - 0.5 * w) * x_factor);
        int top = int((y - 0.5 * h) * y_factor);

        int width = int(w * x_factor);
        int height = int(h * y_factor);

        // std::cout <<scores.size << std::endl;
        // printf("First 5 float values: %.2f %.2f %.2f %.2f %.2f %.2f\n", data[0], data[1], data[2], data[3], data[4], data[5]); // Printing outputs for testing
        std::cout << "Box: " << left << ", " << top << ", " << width << ", " << height << std::endl;


        bboxes.push_back(cv::Rect(left, top, width, height));
      }

      data = base_data + i * dimensions;
    }

    /////////////////////     Apply Non-maximum Suppression to filter out overlapping bounding boxes     ///////////////////// 
    Detection result = applyNMS(detections, bboxes,  confidence, class_ids, classes,  modelScoreThreshold, modelNMSThreshold);
    detections.push_back(result);

    std::cout << "Number of detections: " << detections.size() << std::endl;

    displayResults(frame, detections);

  }

  
  
}