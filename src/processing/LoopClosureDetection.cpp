#include "processing/LoopClosureDetection.h"
#include "util/Config.h"
#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>
#include <DBoW2/TemplatedDatabase.h>
#include <spdlog/spdlog.h>

namespace lightweight_vio {

// DBoW2 type definitions
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> OrbVocabulary;
typedef DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> OrbDatabase;

LoopClosureDetector::LoopClosureDetector() {

    const auto& config = Config::getInstance();
    
    try {
        m_vocabulary = std::make_unique<OrbVocabulary>();
        // CHANGE: Use getter method instead of direct member access
        if (config.loop_closure_enabled() && !config.get_orb_vocabulary_path().empty()) {
            std::cout<<"config.m_orb_vocabulary_path: "<<config.get_orb_vocabulary_path()<<std::endl;
            // CHANGE: Use getter method
            m_vocabulary->loadFromTextFile(config.get_orb_vocabulary_path());
            spdlog::info("[LOOP_CLOSURE] Loaded ORB vocabulary from: {}", config.get_orb_vocabulary_path());
        } else {
            if (!config.loop_closure_enabled()) {
                spdlog::info("[LOOP_CLOSURE] Loop closure detection disabled in config");
            } else {
                spdlog::warn("[LOOP_CLOSURE] No vocabulary path specified in config");
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("[LOOP_CLOSURE] Failed to load vocabulary: {}", e.what());
    }

        // Initialize database with vocabulary - ADD CHECK
    if (m_vocabulary && config.loop_closure_enabled()) {
        m_database = std::make_unique<OrbDatabase>(*m_vocabulary, false, 0);
    }
    
    // Set parameters - CHANGE: Use getter methods
    m_similarity_threshold = config.get_loop_closure_similarity_threshold();
    m_min_loop_interval = config.get_min_loop_interval_frames();
    m_max_database_size = config.get_max_loop_database_size();
}

LoopClosureDetector::~LoopClosureDetector() {
    stop();
}

cv::Mat LoopClosureDetector::stackDescriptors(const std::vector<cv::Mat>& vec)
{
    if (vec.empty()) return cv::Mat();

    cv::Mat mat(vec.size(), vec[0].cols, vec[0].type());
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i].copyTo(mat.row(i));
    }
    return mat;
}

void LoopClosureDetector::start() {
    if (m_thread_running) return;
    
    m_thread_running = true;
    m_loop_closure_thread = std::make_unique<std::thread>(&LoopClosureDetector::loop_closure_thread_function, this);
    spdlog::info("[LOOP_CLOSURE] Loop closure thread started");
}

void LoopClosureDetector::stop() {
    m_thread_running = false;
    m_queue_cv.notify_all();
    
    if (m_loop_closure_thread && m_loop_closure_thread->joinable()) {
        m_loop_closure_thread->join();
    }
    spdlog::info("[LOOP_CLOSURE] Loop closure thread stopped");
}

void LoopClosureDetector::addKeyframe(std::shared_ptr<Frame> keyframe) {
    if (!m_thread_running || !keyframe) return;
    
    std::lock_guard<std::mutex> lock(m_queue_mutex);
    m_keyframe_queue.push(keyframe);
    m_queue_cv.notify_one();
}

void LoopClosureDetector::setLoopCallback(std::function<void(int, int, Eigen::Matrix4f)> callback) {
    m_loop_callback = callback;
}

void LoopClosureDetector::loop_closure_thread_function() {
    while (m_thread_running) {
        std::unique_lock<std::mutex> lock(m_queue_mutex);
        m_queue_cv.wait(lock, [this]() { 
            return !m_thread_running || !m_keyframe_queue.empty(); 
        });
        
        if (!m_thread_running) break;
        
        if (!m_keyframe_queue.empty()) {
            auto keyframe = m_keyframe_queue.front();
            m_keyframe_queue.pop();
            lock.unlock();
            
            processKeyframe(keyframe);
        }
    }
}

void LoopClosureDetector::processKeyframe(std::shared_ptr<Frame> keyframe) {
    const auto& config = Config::getInstance();
    if (!config.loop_closure_enabled() || !m_database || !m_vocabulary) {
        return;
    }
    int current_id = keyframe->get_frame_id();
    
    // Extract ORB features if not already done
    if (m_keyframe_descriptors.find(current_id) == m_keyframe_descriptors.end()) {
        std::vector<cv::Mat> descriptors;
        extractORBFeatures(keyframe, descriptors);
        
        if (descriptors.empty()) {
            spdlog::warn("[LOOP_CLOSURE] No ORB features extracted for keyframe {}", current_id);
            return;
        }
        
        // Store for geometric verification
        m_keyframes[current_id] = keyframe;
        m_keyframe_descriptors[current_id] = descriptors;
        
        // Add to DBoW2 database
        DBoW2::BowVector bow_vec;
        m_vocabulary->transform(descriptors, bow_vec);
        int db_id = m_database->add(bow_vec);
        frameId_to_dbId[current_id] = db_id;
        dbId_to_frameId[db_id] = current_id;
        spdlog::info("[LOOP_CLOSURE] Added keyframe {} to database with {} features", 
                     current_id, descriptors.size());
    }
    
    // Perform loop closure detection
    int loop_candidate_id = -1;
    if (detectLoop(keyframe, loop_candidate_id)) 
    {
        spdlog::info("[LOOP_CLOSURE] Loop candidate detected: {} -> {}", current_id, loop_candidate_id);
        
        // Perform geometric verification
        auto candidate_kf = m_keyframes[loop_candidate_id];
        if (m_keyframe_descriptors.find(loop_candidate_id) == m_keyframe_descriptors.end()) 
        {
            spdlog::error("[LOOP_CLOSURE] Candidate {} was removed from database during processing!", loop_candidate_id);
            return;
        }
        if (m_keyframe_keypoints.find(loop_candidate_id) == m_keyframe_keypoints.end()) 
        {
            spdlog::error("[LOOP_CLOSURE] Candidate {} keypoints were removed from database during processing!", loop_candidate_id);
            return;
        }
        if (geometricVerification(keyframe, candidate_kf)) {
            spdlog::info("[LOOP_CLOSURE] ✅ Loop confirmed: {} -> {}", current_id, loop_candidate_id);
            
            // Calculate relative pose (simplified - you might want more sophisticated pose estimation)
            Eigen::Matrix4f current_pose = keyframe->get_Twb();
            Eigen::Matrix4f candidate_pose = candidate_kf->get_Twb();
            Eigen::Matrix4f relative_pose = candidate_pose.inverse() * current_pose;
            
            // Notify callback
            if (m_loop_callback) {
                m_loop_callback(current_id, loop_candidate_id, relative_pose);
            }
        } else {
            spdlog::info("[LOOP_CLOSURE] ❌ Geometric verification failed for {} -> {}", 
                        current_id, loop_candidate_id);
        }
    }
    else
    {
        spdlog::info("[LOOP_CLOSURE] ❌ loop closure failed for {} -> {}", 
                        current_id, loop_candidate_id);
    }
    
    // Manage database size
    if (m_database->size() > m_max_database_size) 
    {
        int oldest_id = m_keyframes.begin()->first;

        // Clear DB and maps
        m_database->clear();
        frameId_to_dbId.clear();
        dbId_to_frameId.clear();

        // Rebuild without oldest keyframe
        for (auto& [kf_id, kf] : m_keyframes) {
            if (kf_id == oldest_id) continue;

            DBoW2::BowVector bow_vec;
            m_vocabulary->transform(m_keyframe_descriptors[kf_id], bow_vec);
            int new_db_id = m_database->add(bow_vec);

            frameId_to_dbId[kf_id] = new_db_id;
            dbId_to_frameId[new_db_id] = kf_id;
        }

        // Remove oldest from caches
        m_keyframes.erase(oldest_id);
        m_keyframe_descriptors.erase(oldest_id);
        m_keyframe_keypoints.erase(oldest_id);

        spdlog::debug("[LOOP_CLOSURE] Removed and rebuilt without keyframe {}", oldest_id);
    }

}

void LoopClosureDetector::extractORBFeatures(std::shared_ptr<Frame> frame, std::vector<cv::Mat>& descriptors) {
    const auto& config = Config::getInstance();
    if (!config.loop_closure_enabled()) {
        return;
    }
    cv::Mat image;
    if (frame->is_rgbd()) {
        image = frame->get_rgb_image();
    } else {
        image = frame->get_left_image();
    }
    
    if (image.empty()) {
        spdlog::warn("[LOOP_CLOSURE] Empty image for ORB extraction");
        return;
    }
    
    // Convert to grayscale if needed
    cv::Mat gray_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image;
    }
    auto orb = cv::ORB::create(config.get_orb_features(), 
                              config.get_orb_scale_factor(), 
                              config.get_orb_levels(),
                              config.get_orb_edge_threshold(),
                              config.get_orb_first_level(),
                              config.get_orb_wta_k(),
                              cv::ORB::HARRIS_SCORE,
                              config.get_orb_patch_size(),
                              config.get_orb_fast_threshold());
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat orb_descriptors;
    
    // Detect ORB features
    orb->detectAndCompute(gray_image, cv::noArray(), keypoints, orb_descriptors);
    
    if (orb_descriptors.empty()) {
        spdlog::warn("[LOOP_CLOSURE] No ORB features detected");
        return;
    }
    else
    {
        spdlog::info("[LOOP_CLOSURE] {} ORB features detected for keyframe {}", orb_descriptors.rows, frame->get_frame_id());
    }
    
    // Convert descriptors to DBoW2 format
    descriptors.clear();
    for (int i = 0; i < orb_descriptors.rows; ++i) {
        descriptors.push_back(orb_descriptors.row(i));
    }
    
    // Store keypoints for geometric verification
    m_keyframe_keypoints[frame->get_frame_id()] = keypoints;
    spdlog::info("[LOOP_CLOSURE] Stored {} keypoints for frame {}", 
                 keypoints.size(), frame->get_frame_id());
}

bool LoopClosureDetector::detectLoop(std::shared_ptr<Frame> current_kf, int& loop_candidate_id) {
    int current_id = current_kf->get_frame_id();
    spdlog::info("[LOOP_CLOSURE] Loop candidate received in detect loop function: {}", current_id);

    // Query database FIRST
    DBoW2::QueryResults results;
    m_database->query(m_keyframe_descriptors[current_id], results, m_max_database_size);
    
    if (results.empty()) {
        spdlog::info("Keyframe {} returned with 0 loop closure candidates from DBoW2", current_id);
        return false;
    }

    for (const auto& result : results) 
    {
        int candidate_db_id = result.Id;
        int candidate_frame_id = dbId_to_frameId[candidate_db_id];

        long long t_curr_ns = current_kf->get_timestamp();
        long long t_cand_ns = m_keyframes[candidate_frame_id]->get_timestamp();

        double dt = std::abs((t_curr_ns - t_cand_ns) * 1e-9);   // nanoseconds → seconds
        double temporal_threshold = 0.5;
        // TEMPORAL RULE BASED ON TIMESTAMP
        if (dt < temporal_threshold) {
            spdlog::info("[LOOP_CLOSURE] Skipping candidate {} - timestamp diff {:.3f} < {:.3f} sec",
                        candidate_frame_id, dt, temporal_threshold);
            continue;
        }
        else
        {
            spdlog::info("[LOOP_CLOSURE] current {}->candidate {} - similarity_score: {}", current_id,
            candidate_frame_id, result.Score);
 
        }

        // SIMILARITY CHECK
        if (result.Score > m_similarity_threshold) {
            loop_candidate_id = candidate_frame_id;
            spdlog::info("[LOOP_CLOSURE] Candidate {} found with score: {}", 
                        candidate_frame_id, result.Score);
            return true;
        }
    }
    
    spdlog::info("[LOOP_CLOSURE] No suitable candidates found for keyframe {}", current_id);
    return false;
}

bool LoopClosureDetector::geometricVerification(std::shared_ptr<Frame> current_kf, std::shared_ptr<Frame> candidate_kf) {
    int current_id = current_kf->get_frame_id();
    int candidate_id = candidate_kf->get_frame_id();
    
    if (m_keyframe_descriptors.find(current_id) == m_keyframe_descriptors.end() || m_keyframe_descriptors.find(candidate_id) == m_keyframe_descriptors.end()) 
    {
        spdlog::error("[LOOP_CLOSURE] Missing descriptors for geometric verification: {} -> {}", 
                     current_id, candidate_id);
        return false;
    }
    // Get features and keypoints
    auto& current_descriptors = m_keyframe_descriptors[current_id];
    auto& candidate_descriptors = m_keyframe_descriptors[candidate_id];
    auto& current_keypoints = m_keyframe_keypoints[current_id];
    auto& candidate_keypoints = m_keyframe_keypoints[candidate_id];
    
    if (current_descriptors.empty() || candidate_descriptors.empty()) {
        return false;
    }
    
    // Feature matching
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    cv::Mat curr_desc_mat = stackDescriptors(current_descriptors);
    cv::Mat cand_desc_mat = stackDescriptors(candidate_descriptors);

    if (curr_desc_mat.empty() || cand_desc_mat.empty()) {
        spdlog::error("[LOOP_CLOSURE] Empty descriptors during geometric verification");
        return false;
    }

    std::vector<cv::DMatch> matches;
    matcher.match(curr_desc_mat, cand_desc_mat, matches);
    // Filter matches by distance
    double min_dist = std::min_element(matches.begin(), matches.end(),
        [](const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    
    std::vector<cv::DMatch> good_matches;
    for (const auto& match : matches) {
        if (match.distance <= std::max(2.0 * min_dist, 30.0)) {
            good_matches.push_back(match);
        }
    }
    
    if (good_matches.size() < 20) {
        spdlog::debug("[LOOP_CLOSURE] Geometric verification failed: only {} good matches", good_matches.size());
        return false;
    }
    
    // Extract matched points
    std::vector<cv::Point2f> current_points, candidate_points;
    for (const auto& match : good_matches) {
        current_points.push_back(current_keypoints[match.queryIdx].pt);
        candidate_points.push_back(candidate_keypoints[match.trainIdx].pt);
    }
    
    // Find fundamental matrix
    std::vector<uchar> inliers;
    cv::Mat fundamental_matrix = cv::findFundamentalMat(current_points, candidate_points, 
                                                       cv::FM_RANSAC, 3.0, 0.99, inliers);
    
    if (fundamental_matrix.empty()) {
        return false;
    }
    
    // Count inliers
    int inlier_count = cv::countNonZero(inliers);
    double inlier_ratio = static_cast<double>(inlier_count) / good_matches.size();
    
    spdlog::debug("[LOOP_CLOSURE] Geometric verification: {}/{} inliers ({:.2f} ratio)", 
                 inlier_count, good_matches.size(), inlier_ratio);
    
    return inlier_ratio > 0.5; // Threshold for geometric consistency
}

} // namespace lightweight_vio