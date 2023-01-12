/* Copyright 2019-2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example to classify image.
// The input image size must match the input size of the model and be stored as
// RGB pixel array.
// In linux, you may resize and convert an existing image to pixel array like:
//   convert cat.bmp -resize 224x224! cat.rgb
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/classification/adapter.h"
#include "coral/examples/file_utils.h"
#include "coral/tflite_utils.h"
#include "tensorflow/lite/interpreter.h"

ABSL_FLAG(std::string, model_path, "models/050_model.tflite",
          "Path to the tflite model.");
ABSL_FLAG(std::string, image_path, "test_data/sunflower_224.rgb",
          "Path to the image to be classified. The input image size must match "
          "the input size of the model and the image must be stored as RGB "
          "pixel array.");
ABSL_FLAG(std::string, truth_path, "test_data/sunflower_224_truth.txt",
          "Path to the imagenet labels.");
ABSL_FLAG(std::string, log_path, "out/test.log",
          "Path to the log file");

std::vector<float> read_ground_truth( const std::string& file_path) {
    std::ifstream file(file_path.c_str());
  CHECK(file) << "Cannot open " << file_path;
  std::vector<float> data;

    float element;
    while (file >> element)
    {
        data.push_back(element);
    }
    return data;
}

int log(const std::string& file_path, const std::string& status, const std::string& cause) {
    std::ofstream file(file_path.c_str(), std::ios_base::app);
    if (file.is_open())
  {
    auto timenow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char* ctime_no_newline = std::strtok(std::ctime(&timenow), "\n");
    file << ctime_no_newline << "\t" << status << "\t" << cause << "\n";
    file.close();
  }
  else std::cout << "Unable to open file";
  return 0;
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

log(absl::GetFlag(FLAGS_log_path), std::string("OK"), std::string("Starting test")); 

  // Load the model.
  const auto model = coral::LoadModelOrDie(absl::GetFlag(FLAGS_model_path));
  auto edgetpu_context = coral::ContainsEdgeTpuCustomOp(*model)
                             ? coral::GetEdgeTpuContextOrDie()
                             : nullptr;
  auto interpreter =
      coral::MakeEdgeTpuInterpreterOrDie(*model, edgetpu_context.get());
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  CHECK_EQ(interpreter->inputs().size(), 1);
  const auto* input_tensor = interpreter->input_tensor(0);
  CHECK_EQ(input_tensor->type, kTfLiteUInt8)
      << "Only support uint8 input type.";
  auto input = coral::MutableTensorData<uint8_t>(*input_tensor);
  
  coral::ReadFileToOrDie(absl::GetFlag(FLAGS_image_path),
                           reinterpret_cast<char*>(input.data()), input.size());

  CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

    std::vector<float> scores = coral::GetClassificationScores(*interpreter);
    for (int i=0; i < scores.size(); i++) {
        std::cout << scores[i] << std::endl;
    }
    std::vector<float> scores_true = read_ground_truth(absl::GetFlag(FLAGS_truth_path));

    for (int i = 0; i < scores.size(); i++) {
        if (std::abs(scores[i] - scores_true[i]) > 0.001 ) {
           log(absl::GetFlag(FLAGS_log_path), std::string("FAIL"), std::to_string(scores[i]) + " != " + std::to_string(scores_true[i]) + " at index " + std::to_string(i));
           return 1;
        }
    }

    log(absl::GetFlag(FLAGS_log_path), std::string("OK"), std::string("Test passed."));

  return 0;
}
