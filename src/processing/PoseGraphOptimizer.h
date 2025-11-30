#pragma once

#include "database/Frame.h"
#include "database/MapPoint.h"
#include "util/Config.h"
#include <spdlog/spdlog.h>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/robust_kernel_impl.h>

#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>

namespace lightweight_vio {

enum class EdgeType {
    ODOMETRY,
    LOOP_CLOSURE,
};

struct PoseGraph {
    struct Node {
        int id;
        Eigen::Matrix4f pose;  // Twb - world to body transform
        std::shared_ptr<Frame> frame;
        bool fixed = false;
        double timestamp = 0.0;
        
        Node() = default;
        Node(int id, const Eigen::Matrix4f& pose, std::shared_ptr<Frame> frame = nullptr)
            : id(id), pose(pose), frame(frame), fixed(false) {}
    };
    
    struct Edge {
        int from_id, to_id;
        Eigen::Matrix4f measurement;  // T_from_to: from frame to to frame
        Eigen::Matrix<double, 6, 6> information; // Uncertainty in se(3)
        EdgeType type;
        bool enabled = true;
        
        Edge() = default;
        Edge(int from, int to, const Eigen::Matrix4f& meas, EdgeType t)
            : from_id(from), to_id(to), measurement(meas), type(t) {
            information.setIdentity();
        }
    };
    
    std::map<int, Node> nodes;
    std::vector<Edge> edges;
    
    bool hasNode(int id) const { return nodes.count(id) > 0; }
    bool hasEdge(int from_id, int to_id, EdgeType type) const;
    void removeNode(int id);
};

class PoseGraphOptimizer {
private:
    PoseGraph graph_;
    std::mutex graph_mutex_;
    std::mutex optimization_mutex_;
    std::condition_variable optimization_cv_;
    std::atomic<bool> optimization_running_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::unique_ptr<std::thread> optimization_thread_;
    
    // Optimization parameters
    int max_iterations_ = 10;
    double odometry_information_scale_ = 1.0;
    double loop_closure_information_scale_ = 10.0;
    bool use_robust_kernel_ = true;
    double robust_kernel_delta_ = 1.0;
    
public:
    PoseGraphOptimizer();
    ~PoseGraphOptimizer();
    
    // Main interface
    void addKeyframeNode(std::shared_ptr<Frame> frame);
    bool addLoopClosureEdge(int current_id, int candidate_id, 
                           const Eigen::Matrix4f& relative_pose);
    void addOdometryEdge(int new_frame_id);
    bool validateAndFixMeasurement(Eigen::Matrix4f &M, Eigen::Isometry3d &iso_out);
    // Graph management
    void removeKeyframe(int frame_id);
    void clear();
    size_t getNodeCount();
    size_t getEdgeCount();
    
    // Optimization control
    void triggerOptimization();
    void stopOptimization();
    bool isOptimizationRunning() const { return optimization_running_; }
    Eigen::Matrix4f getOptimizedPose(int frame_id);
    std::map<int, Eigen::Matrix4f> getAllOptimizedPoses();
    bool hasOptimizedPose(int frame_id);
    Eigen::Matrix4f getMostRecentOptimizedPose();
    
private:
    // Thread function
    void optimizationThreadFunction();
    
    // Core optimization
    bool optimizeGraph();
    void setupG2OOptimizer(g2o::SparseOptimizer& optimizer);
    void addVerticesToG2O(g2o::SparseOptimizer& optimizer, 
                         std::map<int, g2o::VertexSE3*>& vertex_map);
    void addEdgesToG2O(g2o::SparseOptimizer& optimizer,
                      const std::map<int, g2o::VertexSE3*>& vertex_map);
    void updatePosesFromG2O(const std::map<int, g2o::VertexSE3*>& vertex_map);
    
    // Helper functions
    Eigen::Matrix<double, 6, 6> computeInformationMatrix(EdgeType type) const;
    void correctMapPointsAfterOptimization();
    void updateFramePoses();
    int findPreviousKeyframeId(int current_id) const;
    // Debug and visualization
    void printOptimizationStatistics(const g2o::SparseOptimizer& optimizer) const;
    void logGraphStructure() const;
};

} // namespace lightweight_vio