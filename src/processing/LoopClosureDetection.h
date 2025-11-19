#pragma once

#include "database/Frame.h"
#include "DBoW2/DBoW2.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>

namespace lightweight_vio {

class LoopClosureDetector {
public:
    LoopClosureDetector();
    ~LoopClosureDetector();

    // Add keyframe to queue for processing
    void addKeyframe(std::shared_ptr<Frame> keyframe);

    // Set callback for detected loops
    void setLoopCallback(std::function<void(int, int, Eigen::Matrix4f)> callback);

    // Thread management
    void start();
    void stop();

private:
    void loop_closure_thread_function();
    void processKeyframe(std::shared_ptr<Frame> keyframe);
    bool detectLoop(std::shared_ptr<Frame> current_kf, int& loop_candidate_id);
    bool geometricVerification(std::shared_ptr<Frame> current_kf, std::shared_ptr<Frame> candidate_kf);
    
    // ORB feature extraction for DBoW2
    void extractORBFeatures(std::shared_ptr<Frame> frame, std::vector<cv::Mat>& descriptors);
    
    // DBoW2 components
    std::unique_ptr<OrbVocabulary> m_vocabulary;
    std::unique_ptr<OrbDatabase> m_database;
    
    // Thread-safe queue
    std::queue<std::shared_ptr<Frame>> m_keyframe_queue;
    std::mutex m_queue_mutex;
    std::condition_variable m_queue_cv;
    std::atomic<bool> m_thread_running{false};
    std::unique_ptr<std::thread> m_loop_closure_thread;
    
    // Loop closure parameters
    double m_similarity_threshold;
    int m_min_loop_interval;
    int m_max_database_size;
    
    // Callback for loop detection
    std::function<void(int, int, Eigen::Matrix4f)> m_loop_callback;
    
    // Store keyframe information for geometric verification
    std::unordered_map<int, std::shared_ptr<Frame>> m_keyframes;
    std::unordered_map<int, std::vector<cv::Mat>> m_keyframe_descriptors;
    std::unordered_map<int, std::vector<cv::KeyPoint>> m_keyframe_keypoints;
};

} // namespace lightweight_vio