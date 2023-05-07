#include <fstream>
#include <thread>
#include <condition_variable>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <signal.h>
#include <ctime>
#include <boost/filesystem.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "BYTETracker.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            cerr << "Cuda failure: " << ret << endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.7
#define BBOX_CONF_THRESH 0.1

using namespace nvinfer1;

namespace fs = boost::filesystem;

// stuff we know about the network and the input/output blobs
static const int INPUT_W = 640;
static const int INPUT_H = 640;
const char* INPUT_BLOB_NAME = "input_0";
const char* OUTPUT_BLOB_NAME = "output_0";
static const std::string COL_FILENAME = "Filename";
static const std::string COL_COUNTABLE_ID = "Countable ID";
static const std::string COL_FRAME_NUM = "Frame Num";
static const std::string COL_DIRECTION = "Direction";
static const std::string VAL_LEFT = "Left";
static const std::string VAL_RIGHT = "Right";
static const bool VIDEO_BOXES = false; // Set true to add bounding boxes to videos

static Logger gLogger;

Mat static_resize(Mat& img) {
    float r = min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    Mat re(unpad_h, unpad_w, CV_8UC3);
    resize(img, re, re.size());
    Mat out(INPUT_H, INPUT_W, CV_8UC3, Scalar(114, 114, 114));
    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));
    return out;
}

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static void generate_grids_and_stride(const int target_w, const int target_h, vector<int>& strides, vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const vector<Object>& faceobjects, vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


static void generate_yolox_proposals(vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, vector<Object>& objects)
{
    const int num_class = 1;

    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (num_class + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}

float* blobFromImage(Mat& img){
    cvtColor(img, img, COLOR_BGR2RGB);

    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    vector<float> mean = {0.485, 0.456, 0.406};
    vector<float> std = {0.229, 0.224, 0.225};
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<Vec3b>(h, w)[c]) / 255.0f - mean[c]) / std[c];
            }
        }
    }
    return blob;
}


static void decode_outputs(float* prob, vector<Object>& objects, float scale, const int img_w, const int img_h) {
        vector<Object> proposals;
        vector<int> strides = {8, 16, 32};
        vector<GridAndStride> grid_strides;
        generate_grids_and_stride(INPUT_W, INPUT_H, strides, grid_strides);
        generate_yolox_proposals(grid_strides, prob,  BBOX_CONF_THRESH, proposals);
        //std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);

        vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);


        int count = picked.size();

        //std::cout << "num of boxes: " << count << std::endl;

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x) / scale;
            float y0 = (objects[i].rect.y) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

            // clip
            // x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            // y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            // x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            // y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
}

const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};

static volatile int keepRunning = true;

enum class LastDir {
    NONE,
    RIGHT,
    LEFT
};

void intHandler(int d) {
    keepRunning = false;
}

void doInference(IExecutionContext& context, float* input, float* output, const int output_size, Size input_shape) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void write_csv(const std::vector<std::vector<std::string>>& table, std::ofstream& outfile) {
    for (const auto& row : table) {
        for (auto it = row.begin(); it != row.end(); ++it) {
            outfile << *it;
            if (it != row.end() - 1) {
                outfile << ",";
            }
        }
        outfile << std::endl;
    }
}

void receive_frames(VideoCapture&& cap, const int fps_in, std::queue<Mat>& q_cam, std::queue<float*>& q_blob,
        VideoWriter& writer, std::mutex& mutex_cam, std::condition_variable& cond_cam) {
    Mat img;
	while (keepRunning)
    {
        if(!cap.read(img))
            break;
        
        Mat pr_img = static_resize(img);
        float* blob;
        blob = blobFromImage(pr_img);

        writer.write(img);

        q_cam.push(img);
        q_blob.push(blob);
        cond_cam.notify_one();
    }
    keepRunning = false;
    cond_cam.notify_all();
}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    string encoding_type = "264";
    if (argc >= 4 && string(argv[2]) == "-i") {
        const string engine_file_path {argv[1]};
        ifstream file(engine_file_path, ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
        if (argc >= 7 && string(argv[6]) == "-f") {
            cout << "Using h265 encoding..." << endl;
            encoding_type = "265";
        }
    } else {
        cerr << "arguments not right!" << endl;
        cerr << "run 'python3 tools/trt.py -f exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar' to serialize model first!" << std::endl;
        cerr << "Then use the following command:" << endl;
        cerr << "cd demo/TensorRT/cpp/build" << endl;
        cerr << "./bytetrack ../../../../YOLOX_outputs/yolox_s_mix_det/model_trt.engine -i ../../../../videos/palace.mp4  // deserialize file and run inference" << std::endl;
        cerr << "./bytetrack ../../../../YOLOX_outputs/yolox_s_mix_det/model_trt.engine -i ../../../../videos/palace.mp4 [suffix] [fps]  // deserialize file and run inference" << std::endl;
        return -1;
    }
    const string input_video_path {argv[3]};
    const string output_suffix{argc >= 5 ? argv[4] : "camera"};

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);
    auto output_size = 1;
    for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }
    static float* prob = new float[output_size];

    const string gst_cap_str = "rtspsrc location="+input_video_path+" short-header=TRUE ! rtph"+encoding_type+"depay ! h"+encoding_type+"parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink";
    VideoCapture cap(gst_cap_str, CAP_GSTREAMER);
    //VideoCapture cap(input_video_path);
	if (!cap.isOpened())
		return 0;

	int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    const int fps = argc >= 6 ? std::atoi(argv[5]) : cap.get(CAP_PROP_FPS);
    cout << "fps: " << fps << endl;

    std::string rtsp_prefix = "rtsp";
    const auto check_prefix = std::mismatch(rtsp_prefix.begin(), rtsp_prefix.end(), input_video_path.begin());
    if (check_prefix.first != rtsp_prefix.end()) {
        long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
        cout << "Total frames: " << nFrame << endl;
    }

    auto create_vid_writer = [&](const std::time_t current_time) {
        const auto lt = std::localtime(&current_time);
        char c_timestamp[20];
        std::strftime(c_timestamp, sizeof(c_timestamp), "%Y-%m-%d_%H-%M-%S", lt);
        std::string timestamp(c_timestamp);

        // Save folder: output_suffix/Y-m-d/
        const std::string save_folder = (fs::path(output_suffix) / timestamp.substr(0, timestamp.find("_"))).string();
        fs::create_directories(save_folder);

        const std::string save_path = (fs::path(save_folder) / fs::path(timestamp + "_" + output_suffix + ".mp4")).string();

        cout << "video save_path is " << save_path << endl;

        const auto gst_writer_str = "appsrc ! video/x-raw,format=BGR,width="+to_string(img_w)+",height="+to_string(img_h)+",framerate="+to_string(fps)+"/1 ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! nvv4l2h"+encoding_type+"enc insert-vui=1 ! h"+encoding_type+"parse ! qtmux ! filesink location=" + save_path;
        //VideoWriter writer(save_path, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));
        VideoWriter writer(gst_writer_str, CAP_GSTREAMER, 0, fps, Size(img_w, img_h));


        std::string counts_filename = (fs::path(save_folder) / fs::path(timestamp + "_" + output_suffix + "_counts.csv")).string();
        std::ofstream counts_file(counts_filename);
        cout << "counts save_path is " << counts_filename << endl;

        std::string tracks_filename = (fs::path(save_folder) / fs::path(timestamp + "_" + output_suffix + "_tracks.csv")).string();
        std::ofstream tracks_file(tracks_filename);
        cout << "tracks save_path is " << tracks_filename << endl;

        return std::make_tuple(timestamp, std::move(writer), save_path, std::move(counts_file), std::move(tracks_file));
    };

    auto [timestamp, writer, save_path, counts_file, tracks_file] = create_vid_writer(std::time(nullptr));

    signal(SIGINT, intHandler); // Exit gracefully
    
    /** Counting setup **/
    std::unordered_map<int, std::tuple<int, LastDir>> num_ids;
    // IDs that are countable
    std::vector<std::vector<std::string>> counted_ids;

    // TODO: Needs overhaul of a proper counting algorithm like Line-of-Interest (LoI)
    int hist_thresh = 1;
    double horiz_thresh = 0.9; // Forces rectangular bounding boxes (Rect ratio)
    auto count_thresh = 0.2;

    // Counting thresholds in respective directions in the field of view
    double right_dir_thresh = img_w * count_thresh;
    double left_dir_thresh = img_w * (1.0 - count_thresh);

    bool check_split = false;
    auto start_split_time = chrono::system_clock::now();

    int num_empty = 0;

    std::queue<Mat> q_cam;
    std::queue<float*> q_blob;
    std::mutex mutex_cam;
    std::condition_variable cond_cam;

    std::thread thr_cam(receive_frames, std::move(cap), fps, std::ref(q_cam), std::ref(q_blob), std::ref(writer), std::ref(mutex_cam), std::ref(cond_cam));

    Mat img;
    BYTETracker tracker(fps, 30);
    int num_frames = 0;
    int total_ms = 0;
    int total_ms_true = 0;
    int total_ms_profile = 0;
    int total_ms_before = 0;
    int running_fps = 0;
    int running_fps_true = 0;
	while (keepRunning)
    {
        auto start_true = chrono::system_clock::now();
        auto start_before = chrono::system_clock::now();
        { 
            // Wait for a frame in the queue and get it
            std::unique_lock<std::mutex> lock(mutex_cam);
            cond_cam.wait(lock, [&]{ return !q_cam.empty() || !keepRunning; });
            if (!keepRunning && q_cam.empty()) break;
        }

        img = q_cam.front();
        q_cam.pop();
        auto blob = q_blob.front();
        q_blob.pop();

        num_frames ++;
        if (num_frames % fps == 0)
        {
            counts_file << std::flush;
            tracks_file << std::flush;
            // Split videos every approx. hour
            if (!check_split && (chrono::system_clock::now() - start_split_time) > 
                    chrono::hours(1)) check_split = true;

            const auto elapsed = chrono::system_clock::now() - start_split_time;
            // Recreate writer if error or Split every hour if one second of empty frames - 1:30 max
            if (!tracks_file || (check_split && (num_empty > fps || elapsed >= (chrono::hours(1) + chrono::minutes(30))))) {
                std::this_thread::sleep_for(std::chrono::seconds(1));

                try {
                    counts_file.close();
                    tracks_file.close();
                    std::tie(timestamp, writer, save_path, counts_file, tracks_file) = create_vid_writer(std::time(nullptr));
                    start_split_time = chrono::system_clock::now();
                    check_split = false;
                    num_frames = 0;
                    total_ms = 0;
                    total_ms_true = 0;
                } catch (const fs::filesystem_error& ex) {
                    std::cerr << "File system error: " << ex.what() << endl;
                }
            }

            running_fps = (running_fps + (num_frames / (total_ms / 1000000.0))) / 2;
            running_fps_true = (running_fps_true + (num_frames / (total_ms_true / 1000000.0))) / 2;
            cout << "Processing frame " << num_frames << " (" << running_fps << " inference fps)" << " (" << running_fps_true << " fps)" 
                << " (" << num_frames / (total_ms_profile / 1000000.0)  << " profiling fps)" << " (" << num_frames / (total_ms_before / 1000000.0)  << " before fps)" << endl;
            cout << "Frames left: " << q_cam.size() << endl;
        }
		if (img.empty())
			break;

        num_empty++;

        float scale = min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));

        auto end_before = chrono::system_clock::now();
        total_ms_before = total_ms_before + chrono::duration_cast<chrono::microseconds>(end_before - start_before).count();
        
        // run inference
        auto start = chrono::system_clock::now();
        doInference(*context, blob, prob, output_size, Size(INPUT_H, INPUT_W));
        vector<Object> objects;
        decode_outputs(prob, objects, scale, img_w, img_h);
        vector<STrack> output_stracks = tracker.update(objects);
        auto end = chrono::system_clock::now();
        total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();

        auto start_profile = chrono::system_clock::now();
        for (int i = 0; i < output_stracks.size(); i++)
		{
            num_empty = 0; // A detection exists
			vector<float> tlwh = output_stracks[i].tlwh;
            const auto tid = output_stracks[i].track_id;
            const auto score = output_stracks[i].score;
			bool horizontal = tlwh[2] / tlwh[3] > horiz_thresh;
			if (tlwh[2] * tlwh[3] > 20 && horizontal)
			{
                if (num_ids.count(tid) <= 0) num_ids[tid] = std::make_tuple(1, LastDir::NONE);
                else {
                    auto& tid_count = std::get<0>(num_ids[tid]);
                    auto& tid_ldir = std::get<1>(num_ids[tid]);
                    tid_count++;

                    const auto past_left = tlwh[0] + tlwh[2] < left_dir_thresh;
                    const auto past_right = tlwh[0] < right_dir_thresh;
                    if (past_left || past_right) {
                        const auto countable = tid_count > hist_thresh;
                        if (past_left && tid_ldir != LastDir::LEFT && countable) {
                            counted_ids.push_back({timestamp, std::to_string(num_frames), 
                                    std::to_string(tid), VAL_LEFT});
                            tid_ldir = LastDir::LEFT;
                        } else if (past_right && tid_ldir != LastDir::RIGHT && countable) {
                            counted_ids.push_back({timestamp, std::to_string(num_frames), 
                                    std::to_string(tid), VAL_RIGHT});
                            tid_ldir = LastDir::RIGHT;
                        }
                    }
                }

                if (VIDEO_BOXES) {
                    Scalar s = tracker.get_color(tid);
                    putText(img, format("%d", tid), Point(tlwh[0], tlwh[1] - 5), 
                            0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                    rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
                }
			}

            std::vector<std::vector<std::string>> track{{
                    to_string(num_frames), to_string(tid), 
                    to_string(tlwh[0]), to_string(tlwh[1]), 
                    to_string(tlwh[2]), to_string(tlwh[3]), 
                    to_string(score), "-1", "-1"
                }};
            write_csv(track, tracks_file);
		}
        if (VIDEO_BOXES) {
            putText(img, format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()), 
                    Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        }
        write_csv(counted_ids, counts_file);
        counted_ids.clear();

        delete blob;

        auto end_profile = chrono::system_clock::now();
        total_ms_profile += chrono::duration_cast<chrono::microseconds>(end_profile - start_profile).count();
        auto end_true = chrono::system_clock::now();
        total_ms_true += chrono::duration_cast<chrono::microseconds>(end_true - start_true).count();
    }
    counts_file.close();
    thr_cam.join();
    cout << "FPS: " << running_fps << endl;
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
