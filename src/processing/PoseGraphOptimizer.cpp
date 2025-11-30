#include "processing/PoseGraphOptimizer.h"
#include "util/Config.h"
#include "database/Feature.h"
#include <chrono>
#include <memory>
#include <Eigen/SVD>

namespace lightweight_vio {

// PoseGraph methods
bool PoseGraph::hasEdge(int from_id, int to_id, EdgeType type) const {
    for (const auto& edge : edges) {
        if (edge.from_id == from_id && edge.to_id == to_id && edge.type == type) {
            return true;
        }
    }
    return false;
}

void PoseGraph::removeNode(int id) {
    // Remove node
    nodes.erase(id);
    
    // Remove all edges connected to this node
    edges.erase(
        std::remove_if(edges.begin(), edges.end(),
            [id](const Edge& edge) {
                return edge.from_id == id || edge.to_id == id;
            }),
        edges.end()
    );
}

// PoseGraphOptimizer implementation
PoseGraphOptimizer::PoseGraphOptimizer() {
    // Get parameters from config
    const auto& config = Config::getInstance();
    max_iterations_ = config.m_pgo_max_iterations;
    odometry_information_scale_ = config.m_pgo_odometry_information_scale;
    loop_closure_information_scale_ = config.m_pgo_loop_closure_information_scale;
    use_robust_kernel_ = config.m_pgo_use_robust_kernel;
    robust_kernel_delta_ = config.m_pgo_robust_kernel_delta;
    
    // Start optimization thread
    optimization_thread_ = std::make_unique<std::thread>(
        &PoseGraphOptimizer::optimizationThreadFunction, this);
    
    spdlog::info("[PGO] Pose Graph Optimizer initialized");
}

bool PoseGraphOptimizer::validateAndFixMeasurement(Eigen::Matrix4f &M, Eigen::Isometry3d &iso_out) {
    // Check finite
    if (!M.allFinite()) return false;

    // Extract rotation & translation
    Eigen::Matrix3f Rf = M.block<3,3>(0,0);
    Eigen::Vector3f tf = M.block<3,1>(0,3);

    // Check determinant
    double det = Rf.cast<double>().determinant();
    if (!std::isfinite(det)) return false;

    // Orthonormalize via SVD if det deviates or Rnot orthonormal
    Eigen::Matrix3d R = Rf.cast<double>();
    double orth_err = (R.transpose() * R - Eigen::Matrix3d::Identity()).norm();

    if (std::abs(det - 1.0) > 1e-3 || orth_err > 1e-6) {
        // SVD orthonormalize, enforce det=+1
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        Eigen::Matrix3d Rn = U * V.transpose();
        if (Rn.determinant() < 0) {
            // flip sign on last column of U
            U.col(2) *= -1;
            Rn = U * V.transpose();
        }
        R = Rn;
    }

    Eigen::Quaterniond q(R);
    if (!q.coeffs().allFinite()) return false;
    q.normalize();

    iso_out = Eigen::Isometry3d::Identity();
    iso_out.rotate(q);
    iso_out.pretranslate(tf.cast<double>());

    return true;
}

Eigen::Matrix4f PoseGraphOptimizer::getOptimizedPose(int frame_id) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    auto it = graph_.nodes.find(frame_id);
    if (it != graph_.nodes.end()) {
        return it->second.pose;
    }
    return Eigen::Matrix4f::Identity();
}

std::map<int, Eigen::Matrix4f> PoseGraphOptimizer::getAllOptimizedPoses() {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    std::map<int, Eigen::Matrix4f> optimized_poses;
    for (const auto& [id, node] : graph_.nodes) {
        optimized_poses[id] = node.pose;
    }
    return optimized_poses;
}

bool PoseGraphOptimizer::hasOptimizedPose(int frame_id) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return graph_.nodes.find(frame_id) != graph_.nodes.end();
}

Eigen::Matrix4f PoseGraphOptimizer::getMostRecentOptimizedPose() {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    if (graph_.nodes.empty()) {
        return Eigen::Matrix4f::Identity();
    }
    
    int max_id = -1;
    for (const auto& [id, node] : graph_.nodes) {
        if (id > max_id) {
            max_id = id;
        }
    }
    
    return graph_.nodes.at(max_id).pose;
}

PoseGraphOptimizer::~PoseGraphOptimizer() {
    stopOptimization();
    
    if (optimization_thread_ && optimization_thread_->joinable()) {
        optimization_thread_->join();
    }
}

void PoseGraphOptimizer::addKeyframeNode(std::shared_ptr<Frame> frame) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    int frame_id = frame->get_frame_id();
    
    // Check if node already exists
    if (graph_.hasNode(frame_id)) {
        spdlog::warn("[PGO] Node {} already exists in graph", frame_id);
        return;
    }
    
    PoseGraph::Node node;
    node.id = frame_id;
    node.pose = frame->get_Twb();
    node.frame = frame;
    node.timestamp = static_cast<double>(frame->get_timestamp()) / 1e9;
    
    // Fix first node to eliminate gauge freedom
    if (graph_.nodes.empty()) {
        node.fixed = true;
        spdlog::info("[PGO] First node {} fixed to eliminate gauge freedom", frame_id);
    }
    
    graph_.nodes[frame_id] = node;
    
    // Add odometry edge from previous keyframe
    if (graph_.nodes.size() > 1) {
        addOdometryEdge(frame_id);
    }
    
    spdlog::debug("[PGO] Added keyframe node {} to pose graph", frame_id);
}

void PoseGraphOptimizer::addOdometryEdge(int new_frame_id) {
    int prev_id = findPreviousKeyframeId(new_frame_id);
    if (prev_id == -1) {
        spdlog::warn("[PGO] Could not find previous keyframe for {}", new_frame_id);
        return;
    }
    
    // Get poses
    Eigen::Matrix4f T_prev = graph_.nodes[prev_id].pose;
    Eigen::Matrix4f T_curr = graph_.nodes[new_frame_id].pose;
    
    // Compute relative pose: T_prev_curr = T_prev_world^{-1} * T_curr_world
    Eigen::Matrix4f relative_pose = T_prev.inverse() * T_curr;
    
    PoseGraph::Edge edge(prev_id, new_frame_id, relative_pose, EdgeType::ODOMETRY);
    edge.information = computeInformationMatrix(EdgeType::ODOMETRY);
    
    graph_.edges.push_back(edge);
    
    spdlog::debug("[PGO] Added odometry edge: {} -> {}", prev_id, new_frame_id);
}

bool PoseGraphOptimizer::addLoopClosureEdge(int current_id, int candidate_id, 
                                           const Eigen::Matrix4f& relative_pose) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    // Verify nodes exist
    if (!graph_.hasNode(current_id) || !graph_.hasNode(candidate_id)) {
        spdlog::error("[PGO] Nodes not found for loop closure: {} -> {}", 
                     current_id, candidate_id);
        return false;
    }
    
    // Check if loop closure edge already exists
    if (graph_.hasEdge(candidate_id, current_id, EdgeType::LOOP_CLOSURE)) {
        spdlog::warn("[PGO] Loop closure edge already exists: {} -> {}", 
                    candidate_id, current_id);
        return false;
    }
    
    PoseGraph::Edge edge(candidate_id, current_id, relative_pose, EdgeType::LOOP_CLOSURE);
    edge.information = computeInformationMatrix(EdgeType::LOOP_CLOSURE);
    
    graph_.edges.push_back(edge);
    
    spdlog::info("[PGO] Added loop closure edge: {} -> {}", candidate_id, current_id);
    
    // Trigger optimization
    triggerOptimization();
    return true;
}

void PoseGraphOptimizer::triggerOptimization() {
    if (optimization_running_) {
        spdlog::debug("[PGO] Optimization already running, skipping trigger");
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(optimization_mutex_);
        optimization_running_ = true;
    }
    optimization_cv_.notify_one();
    
    spdlog::info("[PGO] Triggered pose graph optimization");
}

void PoseGraphOptimizer::optimizationThreadFunction() {
    while (!shutdown_requested_) {
        std::unique_lock<std::mutex> lock(optimization_mutex_);
        optimization_cv_.wait(lock, [this]() {
            return optimization_running_ || shutdown_requested_;
        });
        
        if (shutdown_requested_) break;
        
        // Release lock during optimization
        lock.unlock();
        
        // Perform optimization
        auto start_time = std::chrono::high_resolution_clock::now();
        bool success = optimizeGraph();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        if (success) {
            spdlog::info("[PGO] Optimization completed in {} ms", duration.count());
        } else {
            spdlog::warn("[PGO] Optimization failed after {} ms", duration.count());
        }
        
        // Reset flag
        optimization_running_ = false;
    }
}

bool PoseGraphOptimizer::optimizeGraph() {
    std::lock_guard<std::mutex> graph_lock(graph_mutex_);
    
    if (graph_.nodes.size() < 2) {
        spdlog::warn("[PGO] Not enough nodes for optimization: {}", graph_.nodes.size());
        return false;
    }
    
    if (graph_.edges.empty()) {
        spdlog::warn("[PGO] No edges for optimization");
        return false;
    }
    
    try {
        // Create g2o optimizer
        g2o::SparseOptimizer optimizer;
        setupG2OOptimizer(optimizer);
        
        // Add vertices and edges
        std::map<int, g2o::VertexSE3*> vertex_map;
        addVerticesToG2O(optimizer, vertex_map);
        addEdgesToG2O(optimizer, vertex_map);
        
        // Log before optimization
        logGraphStructure();
        
        // Optimize
        optimizer.initializeOptimization();
        int result = optimizer.optimize(max_iterations_);
        
        // Log results
        printOptimizationStatistics(optimizer);
        
        if (result > 0) {
            // Update poses
            updatePosesFromG2O(vertex_map);
            updateFramePoses();
            correctMapPointsAfterOptimization();
            return true;
        } else {
            spdlog::error("[PGO] g2o optimization failed with result: {}", result);
            return false;
        }
        
    } catch (const std::exception& e) {
        spdlog::error("[PGO] Exception during optimization: {}", e.what());
        return false;
    }
}

void PoseGraphOptimizer::setupG2OOptimizer(g2o::SparseOptimizer& optimizer) {
    // Use Eigen linear solver instead of CHOLMOD
    using BlockSolverType = g2o::BlockSolverPL<6, 6>;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto linearSolver = std::make_unique<LinearSolverType>();
    linearSolver->setBlockOrdering(true);

    auto blockSolver = std::make_unique<BlockSolverType>(std::move(linearSolver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    optimizer.setAlgorithm(algorithm);

    if (Config::getInstance().m_enable_debug_output) {
        optimizer.setVerbose(true);
    }
}

void PoseGraphOptimizer::addVerticesToG2O(g2o::SparseOptimizer& optimizer, 
                                         std::map<int, g2o::VertexSE3*>& vertex_map) {
    for (auto& [id, node] : graph_.nodes) {
        auto vertex = new g2o::VertexSE3();
        vertex->setId(id);
        
        // Convert to g2o format (Eigen::Isometry3d)
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.matrix() = node.pose.cast<double>();
        vertex->setEstimate(pose);
        
        vertex->setFixed(node.fixed);
        optimizer.addVertex(vertex);
        vertex_map[id] = vertex;
        
        spdlog::debug("[PGO] Added vertex {} (fixed: {})", id, node.fixed);
    }
}

void PoseGraphOptimizer::addEdgesToG2O(g2o::SparseOptimizer& optimizer,
                                      const std::map<int, g2o::VertexSE3*>& vertex_map) {
    int edge_count = 0;

    auto validateAndFixMeasurement = [](const Eigen::Matrix4f &M_in, Eigen::Isometry3d &iso_out) -> bool {
        if (!M_in.allFinite()) return false;

        Eigen::Matrix3f Rf = M_in.block<3,3>(0,0);
        Eigen::Vector3f tf = M_in.block<3,1>(0,3);

        Eigen::Matrix3d R = Rf.cast<double>();
        double det = R.determinant();
        double orth_err = (R.transpose() * R - Eigen::Matrix3d::Identity()).norm();

        if (!std::isfinite(det)) return false;

        if (std::abs(det - 1.0) > 1e-6 || orth_err > 1e-6) {
            // orthonormalize via SVD
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();
            Eigen::Matrix3d Rn = U * V.transpose();
            if (Rn.determinant() < 0) {
                U.col(2) *= -1;
                Rn = U * V.transpose();
            }
            R = Rn;
        }

        Eigen::Quaterniond q(R);
        if (!q.coeffs().allFinite()) return false;
        q.normalize();

        iso_out = Eigen::Isometry3d::Identity();
        iso_out.rotate(q);
        iso_out.pretranslate(tf.cast<double>());

        return true;
    };

    for (const auto& edge : graph_.edges) {
        if (!edge.enabled) continue;

        // Check if vertices exist
        if (vertex_map.count(edge.from_id) == 0 || vertex_map.count(edge.to_id) == 0) {
            spdlog::warn("[PGO] Skipping edge {}->{}: vertices not found", 
                        edge.from_id, edge.to_id);
            continue;
        }

        // Validate measurement (rotation orthonormality, no NaNs)
        Eigen::Isometry3d meas_iso;
        if (!validateAndFixMeasurement(edge.measurement, meas_iso)) {
            spdlog::warn("[PGO] Skipping edge {}->{}: invalid measurement (NaN/Inf/degenerate rotation)", 
                         edge.from_id, edge.to_id);
            continue;
        }

        // Validate information (symmetric, PD)
        Eigen::Matrix<double,6,6> info = edge.information;
        // Symmetrize
        info = 0.5 * (info + info.transpose());
        // tiny regularization
        info += Eigen::Matrix<double,6,6>::Identity() * 1e-9;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> es(info);
        if (!es.eigenvalues().allFinite() || es.eigenvalues().minCoeff() <= 0) {
            spdlog::warn("[PGO] Skipping edge {}->{}: information matrix not PD (min eig = {})",
                         edge.from_id, edge.to_id, es.eigenvalues().minCoeff());
            continue;
        }

        auto g2o_edge = new g2o::EdgeSE3();
        g2o_edge->setVertex(0, vertex_map.at(edge.from_id));
        g2o_edge->setVertex(1, vertex_map.at(edge.to_id));

        g2o_edge->setMeasurement(meas_iso);
        g2o_edge->setInformation(info);

        if (use_robust_kernel_ && edge.type == EdgeType::LOOP_CLOSURE) {
            auto robust_kernel = new g2o::RobustKernelHuber();
            robust_kernel->setDelta(robust_kernel_delta_);
            g2o_edge->setRobustKernel(robust_kernel);
        }

        optimizer.addEdge(g2o_edge);
        edge_count++;

        spdlog::debug("[PGO] Added edge {}->{} (type: {})", 
                     edge.from_id, edge.to_id, static_cast<int>(edge.type));
    }

    spdlog::info("[PGO] Added {} edges to optimizer", edge_count);
}

// void PoseGraphOptimizer::addEdgesToG2O(g2o::SparseOptimizer& optimizer,
//                                       const std::map<int, g2o::VertexSE3*>& vertex_map) {
//     int edge_count = 0;
    
//     for (auto& edge : graph_.edges) {
//         if (!edge.enabled) continue;
        
//         // Check if vertices exist
//         if (vertex_map.count(edge.from_id) == 0 || vertex_map.count(edge.to_id) == 0) {
//             spdlog::warn("[PGO] Skipping edge {}->{}: vertices not found", 
//                         edge.from_id, edge.to_id);
//             continue;
//         }
        
//         auto g2o_edge = new g2o::EdgeSE3();
//         g2o_edge->setVertex(0, vertex_map.at(edge.from_id));
//         g2o_edge->setVertex(1, vertex_map.at(edge.to_id));
        
//         // Convert measurement
//         Eigen::Isometry3d measurement = Eigen::Isometry3d::Identity();
//         measurement.matrix() = edge.measurement.cast<double>();
//         g2o_edge->setMeasurement(measurement);
        
//         g2o_edge->setInformation(edge.information);
        
//         // Add robust kernel for loop closures to handle outliers
//         if (use_robust_kernel_ && edge.type == EdgeType::LOOP_CLOSURE) {
//             auto robust_kernel = new g2o::RobustKernelHuber();
//             robust_kernel->setDelta(robust_kernel_delta_);
//             g2o_edge->setRobustKernel(robust_kernel);
//         }
        
//         optimizer.addEdge(g2o_edge);
//         edge_count++;
        
//         spdlog::debug("[PGO] Added edge {}->{} (type: {})", 
//                      edge.from_id, edge.to_id, static_cast<int>(edge.type));
//     }
    
//     spdlog::info("[PGO] Added {} edges to optimizer", edge_count);
// }

void PoseGraphOptimizer::updatePosesFromG2O(const std::map<int, g2o::VertexSE3*>& vertex_map) {
    for (auto& [id, node] : graph_.nodes) {
        if (vertex_map.count(id)) {
            Eigen::Isometry3d optimized_pose = vertex_map.at(id)->estimate();
            node.pose = optimized_pose.matrix().cast<float>();
            
            spdlog::debug("[PGO] Updated node {} pose", id);
        }
    }
}

void PoseGraphOptimizer::updateFramePoses() {
    for (auto& [id, node] : graph_.nodes) {
        if (node.frame) {
            node.frame->set_Twb(node.pose);
            spdlog::debug("[PGO] Updated frame {} pose", id);
        }
    }
}

void PoseGraphOptimizer::correctMapPointsAfterOptimization() {
    // Collect all map points from all keyframes
    std::set<std::shared_ptr<MapPoint>> all_map_points;
    
    for (auto& [id, node] : graph_.nodes) {
        if (!node.frame) continue;
        
        const auto& map_points = node.frame->get_map_points();
        for (const auto& mp : map_points) {
            if (mp && !mp->is_bad()) {
                all_map_points.insert(mp);
            }
        }
    }
    
    // Update each map point using its observations
    for (auto& map_point : all_map_points) {
        auto observations = map_point->get_observations();
        if (observations.empty()) continue;
        
        // Use triangulation from all observations to update position
        std::vector<Eigen::Vector3f> world_points;
        
        for (const auto& [frame_ptr, feat_idx] : observations) {
            auto frame = frame_ptr.lock();
            if (!frame) continue;
            
            auto feature = frame->get_feature(feat_idx);
            if (!feature || !feature->is_valid()) continue;
            
            // Get 3D point in camera coordinates
            Eigen::Vector3f camera_point = feature->get_3d_point();
            if (camera_point.isZero()) continue;
            
            // Transform to world using optimized pose
            Eigen::Matrix4f T_wb = frame->get_Twb();
            Eigen::Matrix4f T_cb = frame->get_Tcb().cast<float>();
            
            Eigen::Vector4f camera_homogeneous(camera_point.x(), camera_point.y(), 
                                             camera_point.z(), 1.0f);
            Eigen::Vector4f body_point = T_cb * camera_homogeneous;
            Eigen::Vector4f world_point = T_wb * body_point;
            
            world_points.push_back(world_point.head<3>());
        }
        
        if (!world_points.empty()) {
            // Use median or mean of all observations
            Eigen::Vector3f new_position = Eigen::Vector3f::Zero();
            for (const auto& pos : world_points) {
                new_position += pos;
            }
            new_position /= world_points.size();
            
            map_point->set_position(new_position);
        }
    }
    
    spdlog::info("[PGO] Corrected {} map points", all_map_points.size());
}

Eigen::Matrix<double,6,6> PoseGraphOptimizer::computeInformationMatrix(EdgeType type) const {
    Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Zero();

    double scale = 1.0;
    switch (type) {
        case EdgeType::ODOMETRY:
            scale = odometry_information_scale_;
            // g2o EdgeSE3 uses (translation, rotation) ordering for the 6-vector.
            information.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * (scale);       // translation
            information.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * (scale * 10.0);// rotation
            break;
        case EdgeType::LOOP_CLOSURE:
            scale = loop_closure_information_scale_;
            information.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * (scale * 2.0); // translation
            information.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * (scale * 5.0); // rotation
            break;
    }

    // Ensure symmetry and add tiny diagonal for numerical stability
    information = 0.5 * (information + information.transpose());
    const double eps = 1e-9;
    information += Eigen::Matrix<double,6,6>::Identity() * eps;

    return information;
}

int PoseGraphOptimizer::findPreviousKeyframeId(int current_id) const {
    int prev_id = -1;
    double max_timestamp = -1.0;
    
    for (const auto& [id, node] : graph_.nodes) {
        if (id < current_id && node.timestamp > max_timestamp) {
            max_timestamp = node.timestamp;
            prev_id = id;
        }
    }
    
    return prev_id;
}

void PoseGraphOptimizer::stopOptimization() {
    shutdown_requested_ = true;
    optimization_running_ = false;
    optimization_cv_.notify_all();
}

void PoseGraphOptimizer::removeKeyframe(int frame_id) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    graph_.removeNode(frame_id);
    spdlog::info("[PGO] Removed keyframe {} from pose graph", frame_id);
}

void PoseGraphOptimizer::clear() {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    graph_.nodes.clear();
    graph_.edges.clear();
    spdlog::info("[PGO] Cleared pose graph");
}

size_t PoseGraphOptimizer::getNodeCount() {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return graph_.nodes.size();
}

size_t PoseGraphOptimizer::getEdgeCount() {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return graph_.edges.size();
}

void PoseGraphOptimizer::printOptimizationStatistics(const g2o::SparseOptimizer& optimizer) const {
    if (!Config::getInstance().m_enable_debug_output) return;
    
    spdlog::info("[PGO] Optimization Statistics:");
    spdlog::info("  - Active vertices: {}", optimizer.activeVertices().size());
    spdlog::info("  - Active edges: {}", optimizer.activeEdges().size());
    spdlog::info("  - Chi2: {:.6f}", optimizer.chi2());
}

void PoseGraphOptimizer::logGraphStructure() const {
    if (!Config::getInstance().m_enable_debug_output) return;
    
    spdlog::info("[PGO] Graph Structure:");
    spdlog::info("  - Nodes: {}", graph_.nodes.size());
    
    int odom_edges = 0, loop_edges = 0;
    for (const auto& edge : graph_.edges) {
        switch (edge.type) {
            case EdgeType::ODOMETRY: odom_edges++; break;
            case EdgeType::LOOP_CLOSURE: loop_edges++; break;
        }
    }
    
    spdlog::info("  - Edges: {} total ({} odom, {} loop)", 
                graph_.edges.size(), odom_edges, loop_edges);
}

} // namespace lightweight_vio