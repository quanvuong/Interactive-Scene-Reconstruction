#include "map_processing_node.h"

#include <ros/ros.h>
#include <ros/package.h>

#include <pcl/common/common.h>
#include <tinyxml.h>

#include <iostream>
#include <fstream>

#include "3rd_party/json.hpp"
#include "3rd_party/hungarian.h"

#include "utils.h"
#include "levenberg_marquardt.h"

namespace MapProcessing
{

// Wall object
Obj3D::Ptr wall;

MapProcessingNode::MapProcessingNode(ros::NodeHandle& node_handle)
    : node_handle_(node_handle)
{
    // load rosparam
    node_handle_.param<std::string>("output_folder", output_folder_,
                                    ros::package::getPath("map_proc"));
    instance_segments_path_ = output_folder_ + "/panoptic_segments/";
    instance_id_file_ = instance_segments_path_ + "id.json";
    node_handle_.param<std::string>("cad_database_path", cad_database_path,
                                    ros::package::getPath("map_proc"));
    node_handle_.param<std::string>("cad_id_file", cad_id_file_,
                                    ros::package::getPath("map_proc") + "/cad_models.csv");
    // node_handle_.param<std::string>("ground_truth_prefix", ground_truth_prefix_, "");

    node_handle_.param<bool>("visualize_scene_mesh", visualize_scene_mesh_, false);
    node_handle_.param<bool>("visualize_optimized_alignment", visualize_optimized_alignment_, false);
    node_handle_.param<bool>("visualize_plane_estimation", visualize_plane_estimation_, false);
    node_handle_.param<bool>("visualize_global_regulation", visualize_global_regulation_, false);
    node_handle_.param<bool>("visualize_final_scene", visualize_final_scene_, true);

    node_handle_.param<bool>("if_verbose", if_verbose_, false);
    node_handle_.param<bool>("match_ground_truth", match_ground_truth_, false);
    node_handle_.param<bool>("save_contact_graph", save_contact_graph_, true);
    node_handle_.param<bool>("save_original_cad_pose", save_original_cad_pose_, false);
    node_handle_.param<bool>("save_with_cad_scale", save_with_cad_scale_, false);
    node_handle_.param<bool>("filter_table_above_noisy_point", filter_table_above_noisy_point_, false);
    node_handle_.param<bool>("fix_floor_table_support", fix_floor_table_support_, false);
    node_handle_.param<bool>("match_cad", match_cad_, true);
    node_handle_.param<bool>("match_cabinet", match_cabinet_, false);
    node_handle_.param<bool>("filter_small_objects", filter_small_objects_, false);

    node_handle_.param<bool>("write_refined_bbox", write_refined_bbox_, false);
    node_handle_.param<float>("scale_ratio", scale_ratio_, 1.0);

    node_handle_.param<int>("ground_axis", ground, 2);
    node_handle_.param<int>("k_cad_candidates", k_cad_candidates_, 30);
    node_handle_.param<float>("wall_heuristics", wall_heuristics_, 0.02);
    node_handle_.param<float>("lambda", lambda_, 0.2);

    // Define ground axis
    ground_axis << 0.0, 0.0, 0.0;
    ground_axis(ground) = 1.0;

    // Fill in canonical_base_transforms
    GenerateCanonicalBaseTransforms(canonical_base_transforms);

    // PCL viewer
    viewer = std::make_shared<Viewer> ();
}


void MapProcessingNode::LoadObjectDatabase(std::unordered_map<std::string,
                                           std::vector<ObjCAD::Ptr>>& cad_database)
{
    std::ifstream infile(cad_id_file_);
    std::string line, subline, element;

    std::string cad_dataset = "", cad_id, category_name;
    std::vector<Eigen::Vector4f> planes;
    Eigen::Vector3f aligned_dims;
    float scale = 1.0f;
    Eigen::Matrix4f aligned_transform = Eigen::Matrix4f::Identity();

    // Iterate through each line and split the content
    std::getline(infile, line);  // csv header
    // Check cad_id_file_ format
    bool isPartNetShapeNet = false;
    std::stringstream s(line);
    int i = 0;
    while (std::getline(s, subline, ','))
        i++;
    if (i == 7)
    {
        isPartNetShapeNet = true;
        std::cout << "Loading PartNet and ShapeNet datasets" << std::endl;
    }
    else if (i != 4)
        throw std::domain_error("Unrecognized cad_id_file format");

    while (std::getline(infile, line))
    {
        std::stringstream s(line);
        planes.clear();
        int i = 0;
        while (std::getline(s, subline, '"'))
        {
            if (isPartNetShapeNet)
            {
                switch (i)
                {
                    case 0:  // dataset, cad_id and category_name
                    {
                        std::stringstream ss(subline);
                        if (std::getline(ss, element, ','))
                        {
                            cad_dataset = element;
                            if (if_verbose_)
                                std::cout << "dataset: " << element << std::endl;
                        }
                        if (std::getline(ss, element, ','))
                        {
                            cad_id = element;
                            if (if_verbose_)
                                std::cout << "cad_id " << element << std::endl;
                        }
                        if (std::getline(ss, element, ','))
                        {
                            category_name = element;
                            if (if_verbose_)
                                std::cout << "category_name " << element << std::endl;
                        }
                        break;
                    }
                    case 1:  // transform
                    {
                        std::stringstream ss(subline);
                        float data[16];
                        int j = 0;
                        while (std::getline(ss, element, ','))
                        {
                            data[j] = std::stof(element);
                            j++;
                        }
                        aligned_transform = Eigen::Map<Eigen::Matrix4f>(data).transpose();
                        break;
                    }
                    case 2:  // scale ",0.12345,"
                    {
                        std::stringstream ss(subline);
                        std::getline(ss, element, ',');
                        if (std::getline(ss, element, ','))
                        {
                            scale = std::stof(element);
                            if (if_verbose_)
                                std::cout << "scale: " << scale << std::endl;
                        }
                        break;
                    }
                    case 3:  // aligned_dim
                    {
                        std::stringstream ss(subline);
                        int j = 0;
                        while (std::getline(ss, element, ','))
                        {
                            aligned_dims(j) = std::stof(element);
                            j++;
                        }
                        break;
                    }
                    case 5:  // planes
                    {
                        std::stringstream ss(subline);
                        int j = 0;
                        Eigen::Vector4f plane;
                        while (std::getline(ss, element, ','))
                        {
                            plane(j % 4) = std::stof(element);

                            if (j % 4 == 3)
                                planes.push_back(plane);
                            j++;
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
            else  // original csv format
            {
                switch (i)
                {
                    case 0:  // cad_id and category_name
                    {
                        std::stringstream ss(subline);
                        if (std::getline(ss, element, ','))
                        {
                            cad_id = element;
                            if (if_verbose_)
                                std::cout << "cad_id " << element << std::endl;
                        }
                        if (std::getline(ss, element, ','))
                        {
                            category_name = element;
                            if (if_verbose_)
                                std::cout << "category_name " << element << std::endl;
                        }
                        break;
                    }
                    case 1:  // aligned_dim
                    {
                        std::stringstream ss(subline);
                        int j = 0;
                        while (std::getline(ss, element, ','))
                        {
                            aligned_dims(j) = std::stof(element);
                            j++;
                        }
                        break;
                    }
                    case 3:  // planes
                    {
                        std::stringstream ss(subline);
                        int j = 0;
                        Eigen::Vector4f plane;
                        while (std::getline(ss, element, ','))
                        {
                            // if (j % 4 == 0)
                            // {
                            //     if (if_verbose_)
                            //         std::cout << "new_plane" << std::endl;
                            // }

                            plane(j % 4) = std::stof(element);

                            if (j % 4 == 3)
                                planes.push_back(plane);
                            j++;
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
            i++;
        }

        // Store into ObjCAD
        ObjCAD::Ptr object_cad = std::make_shared<ObjCAD> (cad_dataset, cad_id,
                category_name, planes, aligned_dims, scale);
        object_cad->SetAlignedTransform(aligned_transform);

        // If choose not to fit CADs for cabinets
        if (!match_cabinet_ && (object_cad->category_name == "Cabinet" ))
            continue;

        // Add into database
        auto it = cad_database.find(object_cad->category_name);
        if (it != cad_database.end())
            it->second.push_back(object_cad);
        else
            cad_database.insert({object_cad->category_name, {object_cad}});
    }
    // Close the File
    infile.close();
}


void MapProcessingNode::LoadInstanceSegments(
        const std::unordered_map<std::string, std::vector<ObjCAD::Ptr>>& cad_database,
        std::vector<Obj3D::Ptr>& objects)
{
    std::ifstream infile(instance_id_file_);
    nlohmann::json j;
    infile >> j;

    int instance_label;
    std::string category_name;

    for (const auto& element : j.items())
    {
        // Process instance segments
        category_name = element.value();
        instance_label = std::stoi(element.key());

        // Load geometric model
        pcl::PointCloud<PointTFull>::Ptr cloud (new pcl::PointCloud<PointTFull>);
        float leaf_size = 0.01;
        ReadCloudFromPLY(instance_segments_path_ + std::to_string(instance_label) + ".ply",
                         cloud, true, leaf_size);
        pcl::PolygonMesh::Ptr mesh (new pcl::PolygonMesh);
        ReadMeshFromPLY(instance_segments_path_ + std::to_string(instance_label) + ".ply", mesh);

        // Filter table above noisy points to fix table no CAD model issue
        if (filter_table_above_noisy_point_ && category_name.compare("Table") == 0)
        {
            pcl::ExtractIndices<PointTFull> extract;
            pcl::PointIndices::Ptr table_outliers (new pcl::PointIndices);
            pcl::PointCloud<PointTFull>::Ptr table_no_noise_cloud (new pcl::PointCloud<PointTFull>);

            for (size_t i = 0u; i < cloud->points.size(); ++i)
            {
                Eigen::Vector3d point (cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
                if (point(ground) > 0.85)  // z value above 0.85 is noisy
                    table_outliers->indices.push_back(i);
            }
            extract.setInputCloud(cloud);
            extract.setIndices(table_outliers);
            extract.setNegative(true);
            extract.filter(*table_no_noise_cloud);
            std::cout << "Filtered "
                      << cloud->points.size() - table_no_noise_cloud->points.size()
                      << " noisy table points above z=0.85"
                      << std::endl;

            cloud = table_no_noise_cloud;  // TODO: deallocation?
        }

        // Scale the whole scene if set
        if (scale_ratio_ != 1.0)
        {
            Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
            transform(0, 0) = transform(0, 0) * scale_ratio_;
            transform(1, 1) = transform(1, 1) * scale_ratio_;
            transform(2, 2) = transform(2, 2) * scale_ratio_;
            pcl::transformPointCloud(*cloud, *cloud, transform);
            TransformMeshFull(mesh, transform, mesh);

            makePath(output_folder_ + "/panoptic_segments_scaled_"
                    + std::to_string(scale_ratio_) +"/", 0777);
            WriteMeshToPLY(output_folder_ + "/panoptic_segments_scaled_"
                    + std::to_string(scale_ratio_) +"/" + std::to_string(instance_label) + ".ply", mesh);
        }

        // Store into Obj3D
        Obj3D::Ptr object = std::make_shared<Obj3D> (instance_label, category_name, cloud, mesh);
        if (category_name == "Wall")
            wall = object;

        object->ComputeBox();

        // Prune false detections (small object)
        // If the observed object's diameter is smaller than 0.4 times the mean diameter of
        // the CAD models of same category, skip the observed object
        auto cate_it = cad_database.find(object->category_name);
        if (cate_it != cad_database.end())
        {
            std::vector<ObjCAD::Ptr> cads = cate_it->second;
            int k = 20;
            float diameter = 0.0;
            float min_dim = 0.0;
            for (int i = 0; i < k; i++)
            {
                int r = rand() % cads.size();
                diameter += cads[r]->GetDiameter();
                min_dim += cads[r]->GetDims().minCoeff();
            }
            diameter /= static_cast<float>(k);
            min_dim /= static_cast<float>(k);

            if (filter_small_objects_ && object->GetDiameter() < diameter*0.4)
                continue;
        }

        // Compute planes and potential supporting planes
        // Planes are computed with RANSAC (pcl::SACMODEL_NORMAL_PLANE)
        //      considers abs distance error and surface normal error [0, pi/2]
        // Potential supporting planes are those RANSAC planes
        //      whose normal.dot(ground_axis) > 0.9. ground_axis is upwards
        object->ComputePlanes();
        object->ComputePotentialSupportingPlanes();
        objects.push_back(object);

        // Visualize plane estimation results
        if (visualize_plane_estimation_)
        {
            std::cout << category_name << object->id << std::endl;
            viewer->AddPolygonMeshes({mesh}, {instance_label});
            viewer->AddPlanes(object->GetPlanes());
            viewer->VisualizeOnce({instance_label});
        }
    }
}


void MapProcessingNode::LoadGroundTruthSegments(std::vector<Obj3D::Ptr>& objects)
{
    pcl::PointCloud<PointTLabel>::Ptr GT_label_cloud (new pcl::PointCloud<PointTLabel>);
    ReadCloudFromPLY(ground_truth_prefix_ + ".ply", GT_label_cloud);
    pcl::PointCloud<PointTFull>::Ptr GT_full_cloud (new pcl::PointCloud<PointTFull>);
    ReadCloudFromPLY(ground_truth_prefix_ + ".ply", GT_full_cloud, false);

    if (GT_label_cloud->points.size() != GT_full_cloud->points.size())
        ROS_ERROR("Error reading point clouds. Number of points inconsistent.");

    std::unordered_map<std::string, std::pair<int, pcl::PointCloud<PointTFull>::Ptr>> pc_to_merge;

    TiXmlDocument doc(ground_truth_prefix_ + "_cg.xml");
    bool loadOkay = doc.LoadFile();
    if (loadOkay)
    {
        TiXmlElement *annotation = doc.FirstChildElement("annotation");
        TiXmlNode *label = 0;
        while (label = annotation->IterateChildren(label))
        {
            std::string str = label->ToElement()->Attribute("id");
            int id = std::stoi(str);

            pcl::PointCloud<PointTFull>::Ptr extracted_cloud (new pcl::PointCloud<PointTFull>);
            pcl::PointIndices::Ptr extracted_indices (new pcl::PointIndices);
            ExtractCloudIndicesByLabel(GT_label_cloud, extracted_indices, id);
            ExtractCloudByIndices(GT_full_cloud, extracted_indices, extracted_cloud);

            std::string category_name;
            if (label->ToElement()->Attribute("nyu_class") != NULL)
                category_name = label->ToElement()->Attribute("nyu_class");
            else if (label->ToElement()->Attribute("text") != NULL)
                category_name = label->ToElement()->Attribute("text");
            else
                ROS_ERROR("No annotated semantic labels!");

            if (category_name == "")
                category_name = "Background";

            if (std::find(layout_class.begin(), layout_class.end(), category_name) != layout_class.end())
            {
                auto pc_merge_it = pc_to_merge.find(category_name);
                if (pc_merge_it != pc_to_merge.end())
                    *pc_merge_it->second.second += *extracted_cloud;
                else
                    pc_to_merge.insert(std::make_pair(
                                category_name, std::make_pair(id, extracted_cloud)));

                continue;
            }

            // Store into Obj3D
            Obj3D::Ptr object = std::make_shared<Obj3D> (id, category_name, extracted_cloud);
            object->ComputeBox();
            object->ComputePlanes();
            object->ComputePotentialSupportingPlanes();
            objects.push_back(object);
        }
    }

    for (const auto& layout : pc_to_merge)
    {
        // Store into Obj3D
        Obj3D::Ptr object = std::make_shared<Obj3D> (layout.second.first,
                layout.first, layout.second.second);
        object->ComputeBox();
        object->ComputePlanes();
        object->ComputePotentialSupportingPlanes();
        objects.push_back(object);

        if (object->category_name == "Wall")
            wall = object;
    }
}


void MapProcessingNode::LoadGroundTruthSegments(std::unordered_map<int, OBBox>& gt_objects)
{
    pcl::PointCloud<PointTLabel>::Ptr GT_map_cloud (new pcl::PointCloud<PointTLabel>);
    pcl::io::loadPLYFile(ground_truth_prefix_ + ".ply", *GT_map_cloud);

    TiXmlDocument doc(ground_truth_prefix_ + "_cg.xml");
    bool loadOkay = doc.LoadFile();
    if (loadOkay)
    {
        TiXmlElement *annotation = doc.FirstChildElement("annotation");
        TiXmlNode *label = 0;
        while (label = annotation->IterateChildren(label))
        {
            std::string str = label->ToElement()->Attribute("id");
            int id = std::stoi(str);
            // std::string text_str = label->ToElement()->Attribute("text");
            // if (text_str.size() == 0)
            //     continue;

            pcl::PointCloud<PointTLabel>::Ptr label_cloud (new pcl::PointCloud<PointTLabel>);
            ExtractCloudByLabel(GT_map_cloud, label_cloud, id);
            pcl::PointCloud<PointTFull>::Ptr full_cloud (new pcl::PointCloud<PointTFull>);
            pcl::copyPointCloud(*label_cloud, *full_cloud);

            OBBox box;
            ComputeGroundOrientedBoundingBox(full_cloud, box, ground);
            gt_objects.insert(std::make_pair(id, box));
        }
    }
}


void MapProcessingNode::DecideSupportingParents(const std::vector<Obj3D::Ptr>& objects,
        std::unordered_map<Obj3D::Ptr,
                           std::unordered_map<Obj3D::Ptr, Eigen::Vector4f>>& parent_child_map,
        std::queue<Obj3D::Ptr>& obj_to_check)
{
    // For non-layout object,
    //      parent_candidates are those below object and has a 2D overlapping region
    // For layout object, parent_candidates are {}
    std::unordered_map<Obj3D::Ptr, std::vector<Obj3D::Ptr>> parent_candidates;
    ComputeParentCandidates(objects, parent_candidates);

    for (int i = 0; i < objects.size(); i++)
    {
        auto parent_it = parent_candidates.find(objects[i]);
        if (parent_it != parent_candidates.end())
        {
            float best_score = -1.0;
            Obj3D::Ptr best_parent;
            Eigen::Vector4f best_plane;

            std::vector<Obj3D::Ptr> layout_parents_no_supporting_planes;
            for (int j = 0; j < parent_it->second.size(); j++)
            {
                // when the supporting parent candidate is layout class
                // and without potential supporting planes
                if (parent_it->second[j]->IsLayout() &&
                    parent_it->second[j]->GetPotentialSupportingPlanes().size() == 0)
                {
                    layout_parents_no_supporting_planes.push_back(parent_it->second[j]);
                    continue;
                }

                // Eq. 3 in paper
                std::vector<std::pair<Eigen::Vector4f, float>> plane_score_vec;
                ComputeSupportingScores(objects[i], parent_it->second[j], plane_score_vec);

                // For Table and Floor, set supporting scores to be 1.0
                // so partially observed table can still have floor as its supporting parent
                if (fix_floor_table_support_ &&
                    objects[i]->category_name.compare("Table") == 0 &&
                    parent_it->second[j]->category_name.compare("Floor") == 0)
                {
                    if (plane_score_vec.size() == 0)
                        ROS_ERROR("Floor has no supporting plane for Table");
                    else if (plane_score_vec.size() > 1)
                        ROS_ERROR("Floor has more than one supporting planes for Table");

                    plane_score_vec[0].second = 1.0f;
                    std::cout << "Fixing Floor-Table support with plane ("
                              << plane_score_vec[0].first(0) << ", "
                              << plane_score_vec[0].first(1) << ", "
                              << plane_score_vec[0].first(2) << ", "
                              << plane_score_vec[0].first(3) << ")"
                              << std::endl;
                }

                for (auto score_it = plane_score_vec.begin();
                          score_it != plane_score_vec.end(); score_it++)
                    if (score_it->second > best_score)
                    {
                        // The larger, the better
                        best_score = score_it->second;
                        best_plane = score_it->first;
                        best_parent = parent_it->second[j];
                    }
            }

            // no valid parent, consider layout parents (especially walls and background)
            if (best_score <= 0.2)
            {
                best_score = -1;
                for (const auto& parent : layout_parents_no_supporting_planes)
                {
                    float dist;
                    Eigen::Vector4f hidden_plane(ground_axis(0), ground_axis(1), ground_axis(2),
                                                 -objects[i]->GetBottomHeight());
                    // FIXME: maybe a bug, dist always equals zero. Maybe should be std::max
                    //        Unless it's desired to always accept the first one as parent
                    if (objects[i]->GetMeshPtr() != nullptr)
                        dist = std::min(0.0f,
                                ComputeMeshMeshDistance(objects[i]->GetMeshPtr(),
                                                        parent->GetMeshPtr()));
                    else
                        dist = std::min(0.0f,
                                ComputeCloudCloudDistance(objects[i]->GetPointCloudPtr(),
                                                          parent->GetPointCloudPtr()));
                    float score = (1.0 - std::min(dist, 1.0f)/1.0f);  // always = 1.0

                    if (score > best_score)
                    {
                        best_score = score;
                        best_plane = hidden_plane;
                        best_parent = parent;
                    }
                }
            }

            if (best_score > 0.2)
            {
                if (if_verbose_)
                {
                    std::cout << "child: " << objects[i]->category_name
                        << objects[i]->id << " ";
                    std::cout << "parent: " << best_parent->category_name
                        << best_parent->id << std::endl;
                }
                auto it = parent_child_map.find(best_parent);
                if (it != parent_child_map.end())
                    it->second.insert({objects[i], best_plane});
                else
                    parent_child_map.insert({best_parent, {{objects[i], best_plane}}});
                continue;
            }
        }
        // No parent candidates, nodes connected to root node
        obj_to_check.push(objects[i]);
    }
}


void MapProcessingNode::BuildContactGraph(
        const std::unordered_map<Obj3D::Ptr,
                                 std::unordered_map<Obj3D::Ptr, Eigen::Vector4f>>& parent_child_map,
        const std::unordered_map<int, OBBox>& gt_objects,
        std::queue<Obj3D::Ptr>& obj_to_check,
        pgm::ParseGraphMap::Ptr& contact_graph)
{
    if (contact_graph == nullptr)
    {
        ROS_ERROR("Null contact graph pointer!");
        return;
    }

    // No parent candidates, bottom nodes in the contact graph
    while (!obj_to_check.empty())
    {
        Obj3D::Ptr current_object = obj_to_check.front();
        obj_to_check.pop();  // Remove current object

        // Create and insert node to contact graph
        pgm::ObjectNode::Ptr obj_node = std::make_shared<pgm::ObjectNode>
            (current_object->id, current_object->category_name);

        if (gt_objects.size() > 0)
        {
            // Add matched ground-truth label and iou
            std::vector<std::pair<int, float>> label_iou_vec;
            if (match_ground_truth_)
                label_iou_vec = {{current_object->id, 1.0f}};
            else if (!current_object->IsLayout())
            {
                for (const auto& gt_pair : gt_objects)
                {
                    float iou = ComputeGroundedBoxIOU3D(current_object->GetBox(),
                                                        gt_pair.second, ground);
                    if (iou > 0.0f)
                        label_iou_vec.push_back(std::make_pair(gt_pair.first, iou));
                }
                // Sort it
                auto cmp = [](const auto & a, const auto & b) -> bool {return a.second > b.second;};
                std::sort(label_iou_vec.begin(), label_iou_vec.end(), cmp);
            }
            obj_node->setIoUs(label_iou_vec);
        }

        Obj3D::Ptr parent = current_object->GetSupportingParent().first;
        if (parent == nullptr)
        {
            if (if_verbose_)
                std::cout << "Direct under Room " << current_object->category_name << std::endl;
            contact_graph->push_back(*obj_node);
        }
        else
            contact_graph->insertNode(parent->id, *obj_node);

        // Push child nodes to the queue
        auto parent_it = parent_child_map.find(current_object);
        if (parent_it != parent_child_map.end())
        {
            for (auto child_it = parent_it->second.begin();
                      child_it != parent_it->second.end(); child_it++)
                obj_to_check.push(child_it->first);
        }
    }

    if (if_verbose_)
    {
        // Check parsegraph
        std::vector<pgm::PgEdge> edges = contact_graph->getEdges();
        std::cout << "PG Edges" << std::endl;
        for (const auto& e : edges)
            std::cout << e << std::endl;

        std::vector<pgm::NodeBase::Ptr> nodes = contact_graph->getNodes();
        std::cout << "PG Nodes" << std::endl;
        for (const auto& n : nodes)
            std::cout << n << std::endl;
    }
}


void MapProcessingNode::ComputeParentCandidates(const std::vector<Obj3D::Ptr>& objects,
        std::unordered_map<Obj3D::Ptr, std::vector<Obj3D::Ptr>>& candidates)
{
    for (int i = 0; i < objects.size(); i++)
    {
        // For layout, we don't estimate supporting parents;
        // layout nodes will be directly under root node
        if (objects[i]->IsLayout())
            candidates.insert({objects[i], {}});

        for (int j = i+1; j < objects.size(); j++)
        {
            if (!CheckOverlap2D(objects[i]->GetBox(), objects[j]->GetBox(), ground))
                continue;  // no 2D overlap, cannot provide support

            // objects[j] is below objects[i]
            if (objects[j]->GetBottomHeight() < objects[i]->GetBottomHeight())
            {
                if (!objects[i]->IsLayout())
                {
                    auto it = candidates.find(objects[i]);
                    if (it != candidates.end())
                        it->second.push_back(objects[j]);
                    else
                        candidates.insert({objects[i], {objects[j]}});
                }
            }
            else
            {
                if (!objects[j]->IsLayout())
                {
                    auto it = candidates.find(objects[j]);
                    if (it != candidates.end())
                        it->second.push_back(objects[i]);
                    else
                        candidates.insert({objects[j], {objects[i]}});
                }
            }
        }
    }
}


void MapProcessingNode::ComputeSupportingScores(const Obj3D::Ptr& child,
        const Obj3D::Ptr& parent, std::vector<std::pair<Eigen::Vector4f, float>>& scores)
{
    if (child == nullptr || parent == nullptr)
    {
        ROS_ERROR("Null input object pointer!");
        return;
    }

    // Stuffs can't be supported by things
    if (std::find(layout_class.begin(), layout_class.end(),
                  child->category_name) != layout_class.end() &&
        std::find(layout_class.begin(), layout_class.end(),
                  parent->category_name) == layout_class.end())
        return;

    // Compute overlap ratio and height distance (for each potential supporting planes)
    std::vector<std::pair<Eigen::Vector4f, float>> distances;
    float overlap_ratio = GetOverlapRatio2D(parent->GetBox(), child->GetBox(), ground);
    parent->ComputeSupportDistance(child->GetBottomHeight(), distances);

    // Compute supporting scores from distances
    scores = distances;
    for (int i = 0; i < scores.size(); i++)
        scores[i].second = (1.0 - std::min(distances[i].second, 1.0f)/1.0f) * overlap_ratio;
}


void MapProcessingNode::UpdateObjectsViaSupportingRelations(std::vector<Obj3D::Ptr>& objects,
        std::unordered_map<Obj3D::Ptr,
                           std::unordered_map<Obj3D::Ptr, Eigen::Vector4f>>& parent_child_map,
        const std::queue<Obj3D::Ptr>& obj_to_check)
{
    std::queue<Obj3D::Ptr> next_obj_check = obj_to_check;

    // Top-down refinement
    while (!next_obj_check.empty())
    {
        Obj3D::Ptr current_obj = next_obj_check.front();
        next_obj_check.pop();  // Remove current object
        current_obj->RefineAsSupportingChild();

        // Initialization as parent
        current_obj->ClearInfoAsSupportingParent();
        // Get the children
        auto parent_it = parent_child_map.find(current_obj);
        if (parent_it != parent_child_map.end())
        {
            for (auto child_it = parent_it->second.begin();
                      child_it != parent_it->second.end(); child_it++)
            {
                Obj3D::Ptr child = child_it->first;
                Eigen::Vector4f supporting_plane = child_it->second;
                // Update parents
                current_obj->UpdateAsSupportingParent(child, supporting_plane);
                // Set supporting parent for child
                child->SetSupportingParent(current_obj, supporting_plane);
                // Push into the queue
                next_obj_check.push(child);
            }
            // Remove supporting planes from plane list
            current_obj->UpdatePlanesViaSupporting();
        }
    }

    if (write_refined_bbox_)
    {
        nlohmann::json output_json;
        for (const auto& object : objects)
        {
            nlohmann::json inst_json;
            inst_json["semantic_class"] = object->category_name;

            OBBox box = object->GetBox();
            std::vector<float> box_vec = {box.pos(0), box.pos(1), box.pos(2),
                box.aligned_dims(0), box.aligned_dims(1), box.aligned_dims(2),
                box.quat.x(), box.quat.y(), box.quat.z(), box.quat.w()};
            inst_json["refined_box"] = box_vec;
            output_json[std::to_string(object->id)] = inst_json;
        }
        std::string s = output_json.dump(4);
        std::ofstream out(instance_segments_path_ + "refined_box.json");
        if (out.is_open())
        {
            out << s;
            out.close();
        }
        else
            ROS_ERROR("Cannot open file: ", instance_segments_path_ + "refined_box.json");
    }
}


void MapProcessingNode::MatchCADToSegmentsCoarse(const std::vector<Obj3D::Ptr>& objects,
        const std::unordered_map<std::string, std::vector<ObjCAD::Ptr>>& cad_database,
        std::unordered_map<Obj3D::Ptr, std::vector<ObjCADCandidate::Ptr>>& cad_candidates_map)
{
    for (int i = 0; i < objects.size(); i++)
    {
        std::string category = objects[i]->category_name;

        auto cate_it = cad_database.find(category);
        if (cate_it != cad_database.end())
        {
            ros::Time start = ros::Time::now();

            // Get box, planes, supporting_planes, diameter
            OBBox box = objects[i]->GetBox();
            std::vector<Eigen::Vector4f> planes = objects[i]->GetPlanes();
            std::vector<std::pair<float, Eigen::Vector4f>> supporting_planes =
                objects[i]->GetSupportingPlanes();
            float diameter = objects[i]->GetDiameter();

            // Compute transformed planes (in the local frame, for each canonical pose)
            // world to obj
            Eigen::Matrix4f init_transform = GetHomogeneousTransformMatrix(box.pos, box.quat);
            std::vector<Eigen::Matrix4f> canonical_transforms;
            // world to potential canonical object poses
            GetCanonicalTransforms(canonical_base_transforms, init_transform, canonical_transforms);
            std::vector<std::vector<Eigen::Vector4f>> canonical_transformed_planes;
            // transform all planes into local frame with all canonical transformations
            BatchTransformPlanesGlobalToLocal(planes, canonical_transforms,
                                              canonical_transformed_planes);

            // Compute normalized dimensions, planes (by diameter)
            Eigen::Vector3f normalized_dims = box.aligned_dims / diameter;
            for (auto& planes : canonical_transformed_planes)
                for (auto& plane : planes)
                    plane(3) /= diameter;

            // Closest walls and distance
            Eigen::Vector3f closest_wall_normal;
            float min_dist = 10;
            if (wall != nullptr)
            {
                for (const auto& plane : wall->GetPlanes())
                {
                    Eigen::MatrixXf generalized_corners = objects[i]->GetBoxCorners4D();
                    Eigen::VectorXf distance_array = generalized_corners.transpose() * plane;
                    float dist = distance_array.cwiseAbs().minCoeff();
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        closest_wall_normal = plane.head(3);
                    }
                }
            }

            // Loop over all available CAD models in the same class
            // and compute coarse matching error
            std::vector<ObjCADCandidate::Ptr> cad_candidates;
            for (int j = 0; j < cate_it->second.size(); j++)
            {
                // For each CAD model
                ObjCAD::Ptr cad = cate_it->second[j];
                std::unordered_map<int, float> pose_error_map;
                std::unordered_map<int, std::vector<int>> pose_supporting_plane_map;
                std::unordered_map<int, std::vector<int>> pose_plane_map;

                ComputeDimsMatchingError(normalized_dims, cad, pose_error_map);
                ComputeSupportingPlaneMatchingError(supporting_planes, cad,
                                                    pose_error_map, pose_supporting_plane_map);
                ComputePlaneMatchingError(canonical_transformed_planes, cad,
                                          pose_error_map, pose_plane_map);

                // If there are valid poses for matching, store the maching results
                if (pose_error_map.size() > 0)
                {
                    for (auto& pose_error_pair : pose_error_map)
                    {
                        int pose_index = pose_error_pair.first;
                        // Add heuristic terms to favor poses
                        // s.t. CAD is placed upright (up: +z)
                        //      and facing out against the closest wall (front: -y)
                        pose_error_pair.second += 0.2 * (1.0-ground_axis.transpose()
                                * canonical_transforms[pose_index].col(2).head(3));
                        if (wall != nullptr)
                            pose_error_pair.second += wall_heuristics_
                                * (1.0-closest_wall_normal.transpose()
                                    * (-canonical_transforms[pose_index].col(1).head(3)));

                        // Initialize scale
                        Eigen::Vector3f transformed_dims =
                            (canonical_base_transforms[pose_index].topLeftCorner(3, 3)
                                * cad->GetDims()).cwiseAbs();
                        float scale = box.aligned_dims(ground)/transformed_dims(ground);
                        // Create a object-to-CAD candidate
                        ObjCADCandidate::Ptr cad_candidate = std::make_shared<ObjCADCandidate>(
                                objects[i], cad, pose_index, pose_error_pair.second,
                                pose_supporting_plane_map[pose_index],
                                pose_plane_map[pose_index], scale);
                        cad_candidates.push_back(cad_candidate);
                    }
                }
            }

            // Sort in ascenting order of coarse matching error
            auto cmp = [](const ObjCADCandidate::Ptr & a, const ObjCADCandidate::Ptr & b) -> bool
                {return a->GetCoarseMatchingError() < b->GetCoarseMatchingError();};
            std::sort(cad_candidates.begin(), cad_candidates.end(), cmp);
            cad_candidates_map.insert({objects[i], cad_candidates});

            ros::Time end = ros::Time::now();
            // std::cout << "time: " << (end-start).toSec() << std::endl;

            std::cout << category << objects[i]->id << ": found "
                      << cad_candidates.size() << " CAD candidates" << std::endl;
        }
        else
            std::cout << category << objects[i]->id
                      << ": no CAD model of same category" << std::endl;
    }
}


void MapProcessingNode::ComputeDimsMatchingError(const Eigen::Vector3f& object_normalized_dims,
        const ObjCAD::Ptr& cad, std::unordered_map<int, float>& matching_errors)
{
    if (cad == nullptr)
    {
        ROS_ERROR("Null input cad pointer!");
        return;
    }

    matching_errors.clear();
    for (int i = 0; i < canonical_base_transforms.size(); i++)
    {
        // Assume the cad is placed according to canonical_base_transforms,
        // and we transform the dims to align with the boundingbox
        Eigen::Vector3f transformed_cad_dims =
            (canonical_base_transforms[i].topLeftCorner(3, 3) * cad->GetDims()).cwiseAbs();
        Eigen::Vector3f cad_dim_ratio = transformed_cad_dims / cad->GetDiameter();
        Eigen::Vector3f object_dim_ratio = object_normalized_dims;
        float error = std::abs(cad_dim_ratio(0) - object_dim_ratio(0))
            + std::abs(cad_dim_ratio(1) - object_dim_ratio(1))
            + std::abs(cad_dim_ratio(2) - object_dim_ratio(2));
        matching_errors.insert({i, error});
    }
}


void MapProcessingNode::ComputeSupportingPlaneMatchingError(
        const std::vector<std::pair<float, Eigen::Vector4f>>& supporting_planes,
        const ObjCAD::Ptr& cad, std::unordered_map<int, float>& matching_errors,
        std::unordered_map<int, std::vector<int>>& pose_supporting_plane_map)
{
    if (cad == nullptr)
    {
        ROS_ERROR("Null input cad pointer!");
        return;
    }
    // If object has no supporting plane
    if (supporting_planes.size() == 0)
        return;

    for (auto it = matching_errors.begin(); it != matching_errors.end(); )
    {
        // For each pose
        int index = it->first;
        Eigen::Matrix4f current_transform = canonical_base_transforms[index];
        // Get CAD dims, planes
        Eigen::Vector3f cad_dims = cad->GetDims();
        std::vector<Eigen::Vector4f> cad_planes = cad->GetPlanes();
        // Initialization
        int num_supporting_plane = 0;
        std::vector<std::pair<int, float>> plane_height_ratios;
        // Get potential supporting planes on the CAD
        float ground_dim = (current_transform.topLeftCorner(3, 3) * cad_dims).cwiseAbs()(ground);
        for (int j = 0; j < cad_planes.size(); j++)
        {
            Eigen::Vector4f current_plane = TransformPlaneLocalToGlobal(cad_planes[j],
                                                                        current_transform);
            if (IsSupportingPlane(current_plane, ground_axis))
            {
                num_supporting_plane++;
                float ratio = (-current_plane(3)+ground_dim/2)/ground_dim;
                plane_height_ratios.push_back({j, ratio});
            }
        }
        // Check whether there are sufficient number of
        // potential supporting planes on the CAD for pairing
        // If false, skip this canonical_base_transform
        if (num_supporting_plane < supporting_planes.size())
        {
            it = matching_errors.erase(it);
            continue;
        }
        else
        {
            // Hungarian
            std::vector<std::vector<float>> cost_matrix;
            for (int n = 0; n < supporting_planes.size(); n++)
            {
                std::vector<float> cost_vec;
                for (int m = 0; m < num_supporting_plane; m++)
                {
                    float cost = std::abs(plane_height_ratios[m].second - supporting_planes[n].first);
                    cost_vec.push_back(cost);
                }
                cost_matrix.push_back(cost_vec);
            }

            std::vector<int> obj_to_cad;
            std::vector<int> obj_to_cad_global;
            float supporting_plane_error = MinCostMatchingNonSquare(cost_matrix, obj_to_cad);

            for (int p = 0; p < obj_to_cad.size(); p++)
                obj_to_cad_global.push_back(plane_height_ratios[obj_to_cad[p]].first);

            // Averaged for all planes
            it->second += supporting_plane_error / static_cast<float>(supporting_planes.size());
            pose_supporting_plane_map.insert({index, obj_to_cad_global});
        }
        it++;
    }
}


void MapProcessingNode::ComputePlaneMatchingError(
        const std::vector<std::vector<Eigen::Vector4f>>& canonical_transformed_planes,
        const ObjCAD::Ptr& cad, std::unordered_map<int, float>& matching_errors,
        std::unordered_map<int, std::vector<int>>& pose_plane_map)
{
    if (cad == nullptr)
    {
        ROS_ERROR("Null input cad pointer!");
        return;
    }
    // If object has no plane
    if (canonical_transformed_planes[0].size() == 0)
        return;

    // Loop over all poses
    for (auto it = matching_errors.begin(); it != matching_errors.end(); )
    {
        // For each pose
        int index = it->first;
        std::vector<Eigen::Vector4f> obj_transformed_planes = canonical_transformed_planes[index];
        // Get CAD diameters, planes
        float cad_diameter = cad->GetDiameter();
        std::vector<Eigen::Vector4f> cad_planes = cad->GetPlanes();
        for (auto& plane : cad_planes)
            plane(3) /= cad_diameter;

        // Check whether there are sufficient number of
        // planes on the CAD for pairing
        // If false, skip this canonical_base_transform (which will skip this CAD)
        if (cad_planes.size() < obj_transformed_planes.size())
        {
            it = matching_errors.erase(it);
            continue;
        }
        else
        {
            std::vector<std::vector<float>> cost_matrix;
            for (int n = 0; n < obj_transformed_planes.size(); n++)
            {
                std::vector<float> cost_vec;
                for (int m = 0; m < cad_planes.size(); m++)
                {
                    float cost = ComputePlaneError(cad_planes[m], obj_transformed_planes[n]);
                    cost_vec.push_back(cost);
                }
                cost_matrix.push_back(cost_vec);
            }

            std::vector<int> obj_to_cad;
            float plane_error = MinCostMatchingNonSquare(cost_matrix, obj_to_cad);
            // Averaged for all planes
            it->second += plane_error / static_cast<float>(obj_transformed_planes.size());
            pose_plane_map.insert({index, obj_to_cad});
        }
        it++;
    }
}


bool MapProcessingNode::ValidateSupportedArea(const ObjCADCandidate::Ptr& cad_candidate,
        const float min_valid_area)
{
    if (cad_candidate == nullptr)
    {
        ROS_ERROR("Null input pointer!");
        return false;
    }

    ObjCAD::Ptr cad = cad_candidate->GetCADPtr();
    // Get world-object transform
    Eigen::Matrix4f current_transform = cad_candidate->GetTransform();
    // Load transformed sampled cloud of CAD
    pcl::PointCloud<PointT>::Ptr transformed_cloud = cad_candidate->GetTransformedSampledCloudPtr();

    if (abs(1.0-ground_axis.transpose()
            * canonical_base_transforms[cad_candidate->GetPoseID()].col(2).head(3)) < 0.01)
        return true;
    else
    {
        Eigen::Vector3f transformed_dims =
            (canonical_base_transforms[cad_candidate->GetPoseID()].topLeftCorner(3, 3)
                * cad->GetDims()).cwiseAbs();
        // float bottom_height = cad_candidate->GetObjPtr()->GetBottomHeight();
        float bottom_height = current_transform(ground, 3) - transformed_dims(ground) / 2;

        pcl::ExtractIndices<PointT> extract;
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        pcl::PointCloud<PointT>::Ptr bottom_cloud (new pcl::PointCloud<PointT>);
        for (int j = 0; j < transformed_cloud->points.size(); j += 3)
        {
            Eigen::Vector3f point(transformed_cloud->points[j].x,
                                  transformed_cloud->points[j].y,
                                  transformed_cloud->points[j].z);
            if (point(ground) < bottom_height + 0.01)
                inliers->indices.push_back(j);
        }
        extract.setInputCloud(transformed_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*bottom_cloud);

        Eigen::Vector4f min_point, max_point;
        pcl::getMinMax3D(*bottom_cloud, min_point, max_point);
        float supported_area = std::abs((max_point((ground+1)%3) - min_point((ground+1)%3))
                                      * (max_point((ground+2)%3) - min_point((ground+2)%3)));
        float supported_area_ratio = supported_area
            / (transformed_dims((ground+1)%3)*transformed_dims((ground+2)%3));

        if (supported_area_ratio < min_valid_area)
            return false;
        else
            return true;
    }
}


bool MapProcessingNode::ValidateSupportingAffordance(ObjCADCandidate::Ptr& cad_candidate)
{
    if (cad_candidate == nullptr)
    {
        ROS_ERROR("Null input pointer!");
        return false;
    }

    Obj3D::Ptr object = cad_candidate->GetObjPtr();
    ObjCAD::Ptr cad = cad_candidate->GetCADPtr();
    // Get world-object transform
    Eigen::Matrix4f current_transform = cad_candidate->GetTransform();
    // Supporting planes of mesh object
    std::vector<std::pair<float, Eigen::Vector4f>> supporting_planes = object->GetSupportingPlanes();
    // Load transformed sampled cloud of CAD
    pcl::PointCloud<PointT>::Ptr transformed_cloud = cad_candidate->GetTransformedSampledCloudPtr();

    // cad supporting planes in world frame (with index in all cad planes)
    std::vector<std::pair<int, Eigen::Vector4f>> cad_transformed_supporting_planes;
    // cad supporting plane inliers cloud (with index in all cad planes)
    std::vector<std::pair<int, pcl::PointCloud<PointT>::Ptr>> cad_plane_clouds;
    // cad clouds above supporting planes (with index in all cad planes)
    std::vector<std::pair<int, pcl::PointCloud<PointT>::Ptr>> cad_above_plane_clouds;
    if (supporting_planes.size() > 0)
    {
        // Process for all supporting planes
        for (int j = 0; j < cad->GetPlanes().size(); j++)
        {
            Eigen::Vector4f current_plane = TransformPlaneLocalToGlobal(cad->GetPlanes()[j],
                                                                        current_transform);
            if (IsSupportingPlane(current_plane, ground_axis))
            {
                pcl::ExtractIndices<PointT> extract;
                pcl::PointIndices::Ptr plane_inliers (new pcl::PointIndices);
                pcl::PointIndices::Ptr above_plane_inliers (new pcl::PointIndices);
                pcl::PointCloud<PointT>::Ptr cad_supporting_plane_cloud (new pcl::PointCloud<PointT>);
                pcl::PointCloud<PointT>::Ptr above_supporting_plane_cloud (new pcl::PointCloud<PointT>);

                for (int k = 0; k < transformed_cloud->points.size(); k += 3)
                {
                    Eigen::Vector3f point(transformed_cloud->points[k].x,
                                          transformed_cloud->points[k].y,
                                          transformed_cloud->points[k].z);
                    if ((point(ground) < -current_plane(3) + 0.015) &&
                        (point(ground) > -current_plane(3) - 0.015))
                        plane_inliers->indices.push_back(k);
                    else if (point(ground) > -current_plane(3) + 0.03)
                        above_plane_inliers->indices.push_back(k);
                }

                extract.setInputCloud(transformed_cloud);
                extract.setIndices(plane_inliers);
                extract.setNegative(false);
                extract.filter(*cad_supporting_plane_cloud);

                extract.setInputCloud(transformed_cloud);
                extract.setIndices(above_plane_inliers);
                extract.setNegative(false);
                extract.filter(*above_supporting_plane_cloud);

                cad_transformed_supporting_planes.push_back({j, current_plane});
                cad_plane_clouds.push_back({j, cad_supporting_plane_cloud});
                cad_above_plane_clouds.push_back({j, above_supporting_plane_cloud});
            }
        }

        std::vector<int> supporting_plane_matching_index = cad_candidate->GetSupportingPlaneMatch();
        std::vector<int> plane_matching_index = cad_candidate->GetPlaneMatch();

        if (supporting_planes.size() > cad_transformed_supporting_planes.size())
        {
            ROS_ERROR("CAD supporting planes less than object!");
            return false;
        }

        // Compute Hungarian cost matrix
        std::vector<std::vector<float>> cost_matrix;
        for (int n = 0; n < supporting_planes.size(); n++)
        {
            std::vector<float> cost_vec;
            for (int m = 0; m < cad_transformed_supporting_planes.size(); m++)
            {
                float cost = std::abs(-cad_transformed_supporting_planes[m].second(3)
                                     + object->GetSupportingPlanes()[n].second(3));
                cost_vec.push_back(cost);
            }
            cost_matrix.push_back(cost_vec);
        }

        // Supporting plane affordance-aware optimal matching
        bool valid_match = false;
        std::vector<int> obj_to_cad;
        while (!valid_match)
        {
            // Recompute matching
            float matching_score = MinCostMatchingNonSquare(cost_matrix, obj_to_cad);
            if (matching_score < 10)
                valid_match = true;
            else
            {
                // No valid match for the cad
                valid_match = false;
                break;
            }
            // Validate supporting affordance for each supporting child
            std::unordered_map<int, int> invalid_supporting_plane_match;
            std::set<int> invalid_current_check;
            std::vector<OBBox> boxes;

            for (const auto& child_pair : object->GetSupportingChildren())
            {
                Obj3D::Ptr child = child_pair.first;
                int index = child_pair.second;
                int cad_index = obj_to_cad[index];
                if (invalid_current_check.find(index) == invalid_current_check.end())
                {
                    // If one match is invalid, will not check it again for other supporting children
                    if (!CheckSupportingAffordance(cad_transformed_supporting_planes[cad_index].second,
                                                   cad_plane_clouds[cad_index].second,
                                                   cad_above_plane_clouds[cad_index].second,
                                                   child))
                    {
                        invalid_supporting_plane_match.insert({index, cad_index});
                        invalid_current_check.emplace(index);
                        valid_match = false;
                    }
                }
            }
            // Update cost matrix
            for (auto invalid_it = invalid_supporting_plane_match.begin();
                      invalid_it != invalid_supporting_plane_match.end(); invalid_it++)
                cost_matrix[invalid_it->first][invalid_it->second] = 100.0;
        }

        if (valid_match)
        {
            // If find best match that passes the affordance check, update match
            std::vector<int> obj_to_cad_global;
            for (int p = 0; p < obj_to_cad.size(); p++)
                obj_to_cad_global.push_back(cad_transformed_supporting_planes[obj_to_cad[p]].first);

            cad_candidate->SetSupportingPlaneMatch(obj_to_cad_global);
            return true;
        }
        else
            return false;
    }
    else
        return true;  // If no supporting planes
}

bool MapProcessingNode::CheckSupportingAffordance(const Eigen::Vector4f& supporting_plane,
        const pcl::PointCloud<PointT>::Ptr& plane_cloud,
        const pcl::PointCloud<PointT>::Ptr& above_plane_cloud,
        const Obj3D::Ptr& child)
{
    if (child == nullptr)
    {
        ROS_ERROR("Null input object pointer!");
        return false;
    }

    if (plane_cloud == nullptr || above_plane_cloud == nullptr)
    {
        ROS_ERROR("Null cloud pointer!");
        return false;
    }

    OBBox child_box = child->GetBox();
    float ground_dim = child_box.aligned_dims(ground);
    float bottom_height = -supporting_plane(3);
    Eigen::Matrix3f child_rot = child_box.quat.toRotationMatrix();
    Eigen::Vector2f child_center(child_box.pos((ground+1)%3),
                                 child_box.pos((ground+2)%3));
    Eigen::Vector2f child_u(child_rot((ground+1)%3, (ground+1)%3),
                            child_rot((ground+2)%3, (ground+1)%3));
    Eigen::Vector2f child_v(child_rot((ground+1)%3, (ground+2)%3),
                            child_rot((ground+2)%3, (ground+2)%3));
    Eigen::Vector2f child_half_dim_2d(child_box.aligned_dims((ground+1)%3)/2,
                                      child_box.aligned_dims((ground+2)%3)/2);
    // Work around for children with small ground rectangle
    child_half_dim_2d(0) += 0.01;
    child_half_dim_2d(1) += 0.01;

    // check plane supporting area
    bool valid_supporting_area = false;
    if (plane_cloud->empty())
        return false;

    float max_len_u = 0, min_len_u = 100;
    float max_len_v = 0, min_len_v = 100;
    std::vector<float> len_v_vec;
    for (int i = 0; i < plane_cloud->points.size(); i += 1)
    {
        Eigen::Vector3f point(plane_cloud->points[i].x,
                              plane_cloud->points[i].y,
                              plane_cloud->points[i].z);
        Eigen::Vector2f point_2d(point((ground+1)%3), point((ground+2)%3));
        Eigen::Vector2f vect = point_2d - child_center;
        float len_u = vect.transpose()*child_u;
        float len_v = vect.transpose()*child_v;
        if ((len_u < child_half_dim_2d(0)) && (len_u > -child_half_dim_2d(0)) &&
            (len_v < child_half_dim_2d(1)) && (len_v > -child_half_dim_2d(1)))
        {
            if (len_u > max_len_u)
                max_len_u = len_u;
            if (len_u < min_len_u)
                min_len_u = len_u;
            if (len_v > max_len_v)
                max_len_v = len_v;
            if (len_v < min_len_v)
                min_len_v = len_v;
        }
    }

    // Second condition is meaningless
    if ((max_len_u > min_len_u) && (max_len_v > min_len_v))
        if ((max_len_u - min_len_u)*(max_len_v - min_len_v) /
            (child_half_dim_2d(0)*child_half_dim_2d(1)*4.0) > 0.0)
            valid_supporting_area = true;

    // check collision
    bool valid_supporting_space = true;
    if (!above_plane_cloud->empty())
    {
        for (int i = 0; i < above_plane_cloud->points.size(); i += 1)
        {
            Eigen::Vector3f point(above_plane_cloud->points[i].x,
                                  above_plane_cloud->points[i].y,
                                  above_plane_cloud->points[i].z);
            if (point(ground) - bottom_height < ground_dim)
            {
                Eigen::Vector2f point_2d(point((ground+1)%3), point((ground+2)%3));
                Eigen::Vector2f vect = point_2d - child_center;
                float len_u = vect.transpose()*child_u;
                float len_v = vect.transpose()*child_v;
                if ((len_u < child_half_dim_2d(0)) && (len_u > -child_half_dim_2d(0)) &&
                    (len_v < child_half_dim_2d(1)) && (len_v > -child_half_dim_2d(1)))
                {
                    valid_supporting_space = false;
                    break;
                }
            }
        }
    }
    return (valid_supporting_area && valid_supporting_space);
}


void MapProcessingNode::ComputePotentialCollisionPairs(
        const std::unordered_map<Obj3D::Ptr,
                                 std::vector<ObjCADCandidate::Ptr>>& cad_selected_candidates,
        std::unordered_map<Obj3D::Ptr,
                           std::unordered_map<Obj3D::Ptr, float>>& possible_collision_map)
{
    std::vector<Obj3D::Ptr> objects;
    for (const auto& cad_candidate_pair : cad_selected_candidates)
        objects.push_back(cad_candidate_pair.first);

    for (const auto& object_1 : objects)
    {
        for (const auto& object_2 : objects)
        {
            std::unordered_map<Obj3D::Ptr, int> children = object_1->GetSupportingChildren();
            if (object_2 != object_1 &&
                object_2 != object_1->GetSupportingParent().first &&
                children.find(object_2) == children.end())
            {
                if (CheckOverlap2DRough(object_1->GetBox(), object_2->GetBox(), ground))
                {
                    if (if_verbose_)
                        std::cout << "Potential collision: " << object_1->category_name << " "
                                  << object_1->id << " "
                                  << object_2->category_name << " "
                                  << object_2->id << std::endl;
                    float overlap_ratio = GetOverlapRatio2D(object_1->GetBox(), object_2->GetBox(),
                                                            ground, 0.005);
                    auto map_it = possible_collision_map.find(object_1);
                    if (map_it != possible_collision_map.end())
                        map_it->second.insert({object_2, overlap_ratio});
                    else
                        possible_collision_map.insert({object_1, {{object_2, overlap_ratio}}});
                }
            }
        }
    }
}


void MapProcessingNode::MatchCADToSegmentsFine(const std::vector<Obj3D::Ptr>& objects,
        std::unordered_map<Obj3D::Ptr,
                           std::vector<ObjCADCandidate::Ptr>>& cad_selected_candidates)
{
    for (auto& obj_candidates_pair : cad_selected_candidates)
    {
        Obj3D::Ptr object = obj_candidates_pair.first;
        Eigen::Matrix4f init_transform = object->GetBoxTransform();
        std::cout << object->category_name << object->id << std::endl;

        for (auto& candidate : obj_candidates_pair.second)
        {
            ObjCAD::Ptr cad = candidate->GetCADPtr();
            int pose_index = candidate->GetPoseID();
            Eigen::Vector3f transformed_dims =
                (canonical_base_transforms[pose_index].topLeftCorner(3, 3)
                    * cad->GetDims()).cwiseAbs();
            std::vector<int> supporting_plane_matching_index = candidate->GetSupportingPlaneMatch();
            std::vector<int> plane_matching_index = candidate->GetPlaneMatch();

            Eigen::Vector3f obj_dims = object->GetBox().aligned_dims;
            std::vector<Eigen::Vector4f> obj_planes = object->GetPlanes();
            std::vector<std::pair<float, Eigen::Vector4f>> obj_supporting_planes
                = object->GetSupportingPlanes();
            std::vector<Eigen::Vector4f> cad_planes = cad->GetPlanes();

            //// Fill in the error function for LM optimization
            FineAlignmentError error_functor;

            // Aligned half dimensions
            error_functor.ground << (ground+1)%3, (ground+2)%3, ground;
            error_functor.half_ground_dim_object = obj_dims(ground)/2;
            error_functor.half_ground_dim_cad = transformed_dims(ground)/2;
            error_functor.half_u_dim_object = obj_dims((ground+1)%3)/2;
            error_functor.half_u_dim_cad = transformed_dims((ground+1)%3)/2;
            error_functor.half_v_dim_object = obj_dims((ground+2)%3)/2;
            error_functor.half_v_dim_cad = transformed_dims((ground+2)%3)/2;

            // Height of supporting planes relative to the supported plane
            int m = obj_supporting_planes.size();
            Eigen::VectorXf supporting_planes_cad(m);
            Eigen::VectorXf supporting_planes_object(m);
            for (int k = 0; k < m; k++)
            {
                // Object plane height
                supporting_planes_object(k) = obj_supporting_planes[k].first * obj_dims(ground);
                Eigen::Vector4f assigned_cad_plane = cad_planes[supporting_plane_matching_index[k]];
                Eigen::Vector4f transformed_supporting_plane =
                    TransformPlaneLocalToGlobal(assigned_cad_plane,
                                                canonical_base_transforms[pose_index]);
                if (IsSupportingPlane(transformed_supporting_plane, ground_axis))
                    // CAD plane height
                    supporting_planes_cad(k) = -transformed_supporting_plane(3)
                                              + transformed_dims(ground)/2;
                else
                    ROS_ERROR("Wrong assignment or transformation of supporting planes!");
            }
            error_functor.m = m;
            error_functor.supporting_planes_cad = supporting_planes_cad;
            error_functor.supporting_planes_object = supporting_planes_object;

            // Planes relative to the bounding box frame (local)
            int n = obj_planes.size();
            Eigen::MatrixXf planes_cad(4, n);
            Eigen::MatrixXf planes_object(4, n);
            for (int k = 0; k < n; k++)
            {
                planes_object.col(k) = TransformPlaneGlobalToLocal(obj_planes[k], init_transform);
                Eigen::Vector4f assigned_cad_plane = cad_planes[plane_matching_index[k]];
                planes_cad.col(k) =
                    TransformPlaneLocalToGlobal(assigned_cad_plane,
                                                canonical_base_transforms[pose_index]);
            }
            error_functor.n = n;
            error_functor.planes_cad = planes_cad;
            error_functor.planes_object = planes_object;
            error_functor.object_initial_box = object->GetBox();

            int w = 0;
            if (wall != nullptr)
            {
                std::vector<Eigen::Vector4f> planes = wall->GetPlanes();
                Eigen::MatrixXf planes_wall(4, planes.size());
                for (int k = 0; k < w; k++)
                    planes_wall.col(k) = planes[k];

                error_functor.planes_wall = planes_wall;
            }
            error_functor.w = w;
            error_functor.N = m + 2*n + w + 10;

            // Parameters to be optimized and their initial value
            error_functor.p = 4;
            Eigen::VectorXf x_0(error_functor.p);
            x_0(0) = candidate->GetScale();             // initial value for 's'
            x_0(1) = 0.0;             // initial value for 'd_theta'
            x_0(2) = 0.0;             // initial value for 'd_u'
            x_0(3) = 0.0;             // initial value for 'd_v'

            Eigen::VectorXf x = x_0;
            Eigen::LevenbergMarquardt<FineAlignmentError, float> lm(error_functor);
            lm.parameters.maxfev = 1000;
            int status = lm.minimize(x);

            Eigen::VectorXf result;
            error_functor(x, result);
            float residual = result.transpose() * result;

            Eigen::Matrix4f refine_transform = GetRefineTransform(x, obj_dims(ground),
                                                                  transformed_dims(ground));
            // Eigen::Matrix4f final_transform = init_transform * refine_transform_no_scale
            //                                 * canonical_base_transforms[pose_index];
            candidate->SetScale(x(0));
            candidate->SetRefinedTransform(refine_transform);
            float ae = ComputeAlignmentError(object, candidate);
            // Also consider coarse matching error
            ae += lambda_ * candidate->GetCoarseMatchingError();
            candidate->SetFineMatchingError(ae);
        }
        auto cmp = [](const ObjCADCandidate::Ptr & a, const ObjCADCandidate::Ptr & b) -> bool
            {return a->GetFineMatchingError() < b->GetFineMatchingError();};
        std::sort(obj_candidates_pair.second.begin(), obj_candidates_pair.second.end(), cmp);

        if (visualize_optimized_alignment_)
        {
            for (int j = 0; j < obj_candidates_pair.second.size(); j++)
            {
                std::vector<OBBox> children_boxes;
                for (auto& children : object->GetSupportingChildren())
                {
                    children_boxes.push_back(children.first->GetBox());
                }

                pcl::PolygonMesh::Ptr transformed_mesh =
                    obj_candidates_pair.second[j]->GetTransformedMeshPtr();
                viewer->AddPolygonMeshes({transformed_mesh, object->GetMeshPtr()}, {object->id, -1});

                children_boxes.push_back(object->GetBox());
                std::cout << "Matching score: "
                          << obj_candidates_pair.second[j]->GetFineMatchingError() << std::endl;

                viewer->VisualizeOnce({object->id, -1});

                // Only visualize first 10 candidates
                if (j > 10)
                    break;
            }
        }
    }
}


float MapProcessingNode::ComputeAlignmentError(const Obj3D::Ptr& object,
        ObjCADCandidate::Ptr& cad_candidate)
{
    if (object == nullptr || cad_candidate == nullptr)
    {
        ROS_ERROR("Null input object pointer!");
        return 100;
    }

    // Transformed cad mesh
    pcl::PolygonMesh::Ptr mesh = cad_candidate->GetTransformedMeshPtr();

    float align_error = ComputePointCloudToMeshError(*(object->GetPointCloudPtr()), *mesh);
    return align_error;
}


void MapProcessingNode::GlobalRegulation(const std::vector<Obj3D::Ptr>& objects,
        std::unordered_map<Obj3D::Ptr,
                           std::vector<ObjCADCandidate::Ptr>>& cad_selected_candidates,
        const std::unordered_map<Obj3D::Ptr,
                                 std::unordered_map<Obj3D::Ptr, float>>& possible_collision_map,
        std::unordered_map<Obj3D::Ptr, ObjCADCandidate::Ptr>& map_candidate)
{
    bool if_wall_constraint = false;
    if (wall != nullptr)
        if_wall_constraint = true;

    // Check object-wall proximal (collision) and supporting constraints
    for (auto& obj_candidates_pair : cad_selected_candidates)
    {
        for (int i = 0; i < obj_candidates_pair.second.size(); )
        {
            ObjCADCandidate::Ptr candidate = obj_candidates_pair.second[i];
            ObjCAD::Ptr cad = candidate->GetCADPtr();
            // Get box for aligned CAD
            OBBox current_box = candidate->GetAlignedBox();

            // Check supporting relations (2D overlap)
            bool valid_supporting_ratio = true;
            std::unordered_map<Obj3D::Ptr, int> children =
                obj_candidates_pair.first->GetSupportingChildren();
            for (const auto& child : children)
            {
                OBBox child_box = child.first->GetBox();
                if (GetOverlapRatio2D(current_box, child_box, ground, 0.002) < 0.3)
                {
                    valid_supporting_ratio = false;
                    break;
                }
            }
            if (!valid_supporting_ratio)
            {
                obj_candidates_pair.second.erase(obj_candidates_pair.second.begin()+i);
                continue;
            }

            // Check collision with walls
            if (if_wall_constraint)
            {
                if (!ValidateWallConstraint (wall->GetPlanes(), obj_candidates_pair.second[i]))
                {
                    obj_candidates_pair.second.erase(obj_candidates_pair.second.begin()+i);
                    continue;
                }
            }
            i++;
        }
        if (obj_candidates_pair.second.size() > 0)
            map_candidate.insert({obj_candidates_pair.first, obj_candidates_pair.second[0]});
        else
        {
            ROS_ERROR("No candidates valid!");
            std::cout << obj_candidates_pair.first->category_name << " "
                      << obj_candidates_pair.first->id << std::endl;
        }
    }

    std::unordered_map<Obj3D::Ptr, std::vector<Obj3D::Ptr>> collided_pairs;
    std::set<Obj3D::Ptr> collided_objects;
    ComputeCollision(map_candidate, possible_collision_map, collided_pairs, collided_objects);
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Collision pairs: " << std::endl;
    for (const auto& collided_pair : collided_pairs)
        for (const auto& collided_obj : collided_pair.second)
            std::cout << collided_pair.first->category_name << " "
                      << collided_pair.first->id << " "
                      << collided_obj->category_name << " "
                      << collided_obj->id << std::endl;
    std::cout << "Collided objects: " << std::endl;
    for (const auto& collided_object : collided_objects)
        std::cout << collided_object->category_name << " "
                  << collided_object->id << std::endl;

    int iter = 0;
    int max_iter = 10;
    while (collided_objects.size() > 0)
    {
        if (visualize_global_regulation_)
        {
            std::vector<OBBox> objects_viz;
            std::vector<pcl::PolygonMesh::Ptr> meshes;
            std::vector<int> id;
            for (int i = 0; i < objects.size(); i++)
            {
                auto obj_it = map_candidate.find(objects[i]);
                if (obj_it != map_candidate.end())
                {
                    ObjCADCandidate::Ptr candidate = obj_it->second;
                    pcl::PolygonMesh::Ptr mesh = candidate->GetTransformedMeshPtr();
                    OBBox current_box = candidate->GetAlignedBox();
                    meshes.push_back(mesh);
                    objects_viz.push_back(current_box);
                }
                else
                {
                    meshes.push_back(objects[i]->GetMeshPtr());
                    objects_viz.push_back(objects[i]->GetBox());
                }
                id.push_back(objects[i]->id);
            }
            viewer->AddPolygonMeshes(meshes, id);
            viewer->AddCubes(objects_viz);
            viewer->VisualizeOnce(id);
        }

        // Solve CSP
        // Randomly select a collided object and try other cad candidates
        int r = rand() % collided_objects.size();
        auto r_it = collided_objects.begin();
        std::advance(r_it, r);
        Obj3D::Ptr current_object = *r_it;
        std::vector<ObjCADCandidate::Ptr> candidates =
            cad_selected_candidates.find(current_object)->second;

        std::cout << "---------------------------------" << std::endl;
        std::cout << "Iter: " << iter << std::endl;
        std::cout << "Choose object: " << current_object->category_name << " "
                  << current_object->id << std::endl;
        std::cout << candidates.size() << std::endl;

        std::vector<Obj3D::Ptr> best_collided_objects;
        ObjCADCandidate::Ptr best_candidate;
        float min_collision_penetration = 10.0f;
        for (int i = 0; i < candidates.size(); i++)
        {
            std::vector<Obj3D::Ptr> current_collided_objects;
            float collision_penetration =
                RecomputeCollision(current_object, candidates[i], map_candidate,
                                   possible_collision_map.find(current_object)->second,
                                   current_collided_objects);
            if (collision_penetration < min_collision_penetration)
            {
                min_collision_penetration = collision_penetration;
                best_collided_objects = current_collided_objects;
                best_candidate = candidates[i];
            }
        }
        // Update map_candidate, collided_pairs, collided_objects
        if (best_candidate != nullptr)
        {
            map_candidate.find(current_object)->second = best_candidate;
            for (auto& collided_pair : collided_pairs)
            {
                if (collided_pair.first == current_object)
                    collided_pair.second = best_collided_objects;
                else if (std::find(best_collided_objects.begin(),
                                   best_collided_objects.end(),
                                   collided_pair.first) != best_collided_objects.end())
                {
                    if (std::find(collided_pair.second.begin(),
                                  collided_pair.second.end(),
                                  current_object) == collided_pair.second.end())
                        collided_pair.second.push_back(current_object);
                }
                else
                {
                    auto it = std::find(collided_pair.second.begin(),
                                        collided_pair.second.end(),
                                        current_object);
                    if (it != collided_pair.second.end())
                        collided_pair.second.erase(it);
                }
            }
            collided_objects.clear();
            for (const auto& collided_pair : collided_pairs)
            {
                if (collided_pair.second.size() > 0)
                    collided_objects.emplace(collided_pair.first);
                for (const auto& collided_obj : collided_pair.second)
                    collided_objects.emplace(collided_obj);
            }
        }

        std::cout << "Collision pairs: " << std::endl;
        for (const auto& collided_pair : collided_pairs)
            for (const auto& collided_obj : collided_pair.second)
                std::cout << collided_pair.first->category_name << " "
                          << collided_pair.first->id << " "
                          << collided_obj->category_name << " "
                          << collided_obj->id << std::endl;
        std::cout << "Collided objects: " << std::endl;
        for (const auto& collided_object : collided_objects)
            std::cout << collided_object->category_name << " "
                      << collided_object->id << std::endl;

        iter++;
        if (iter > max_iter)
            break;
    }
}


bool MapProcessingNode::ValidateWallConstraint(const std::vector<Eigen::Vector4f>& wall_planes,
        const ObjCADCandidate::Ptr& cad_candidate)
{
    for (int i = 0; i < wall_planes.size(); i++)
    {
        Eigen::VectorXf distance_array = cad_candidate->GetAlignedBoxCorners4D().transpose()
                                         * wall_planes[i];
        if ((distance_array.maxCoeff() * distance_array.minCoeff() < 0) &&
            std::min(distance_array.maxCoeff(), -distance_array.minCoeff()) > 0.15)
            return false;
    }
    return true;
}


void MapProcessingNode::ComputeCollision(
        const std::unordered_map<Obj3D::Ptr, ObjCADCandidate::Ptr>& map_candidate,
        const std::unordered_map<Obj3D::Ptr,
                                 std::unordered_map<Obj3D::Ptr, float>>& possible_collision_map,
        std::unordered_map<Obj3D::Ptr, std::vector<Obj3D::Ptr>>& collided_pairs,
        std::set<Obj3D::Ptr>& collided_objects)
{
    ROS_INFO("Start collision check!");
    // Loop over all map objects and check collision based on the possible_collision_map
    for (const auto& obj_candidate : map_candidate)
    {
        Obj3D::Ptr object = obj_candidate.first;
        auto it = possible_collision_map.find(object);
        if (it != possible_collision_map.end())
        {
            pcl::PolygonMesh::Ptr candidate_mesh_1 = obj_candidate.second->GetTransformedMeshPtr();
            for (const auto& collided : it->second)
            {
                Obj3D::Ptr object_to_check = collided.first;
                auto check_it = map_candidate.find(object_to_check);
                if (check_it == map_candidate.end())
                    continue;

                pcl::PolygonMesh::Ptr candidate_mesh_2 = check_it->second->GetTransformedMeshPtr();

                std::vector<Eigen::Vector3f> contact_locations;
                std::vector<float> penetration_depths;

                ros::Time start = ros::Time::now();

                if (ComputeMeshMeshCollision(candidate_mesh_1, candidate_mesh_2,
                                             contact_locations, penetration_depths))
                {
                    auto collide_it = collided_pairs.find(object);
                    if (collide_it != collided_pairs.end())
                        collide_it->second.push_back(object_to_check);
                    else
                        collided_pairs.insert({object, {object_to_check}});
                    collided_objects.emplace(object);
                    collided_objects.emplace(object_to_check);
                }
                ros::Time end = ros::Time::now();
                // std::cout << "Mesh-mesh: " << (end-start).toSec() << std::endl;
            }
        }
    }
}


float MapProcessingNode::RecomputeCollision(const Obj3D::Ptr& object,
        const ObjCADCandidate::Ptr& cad_candidate,
        const std::unordered_map<Obj3D::Ptr, ObjCADCandidate::Ptr>& map_candidate,
        const std::unordered_map<Obj3D::Ptr, float>& possible_collided_objects,
        std::vector<Obj3D::Ptr>& collided_objects)
{
    float total_penetration = 0.0f;
    for (const auto& collide_to_check : possible_collided_objects)
    {
        auto obj_it = map_candidate.find(collide_to_check.first);
        if (obj_it != map_candidate.end())
        {
            // Prepare transformed mesh
            pcl::PolygonMesh::Ptr candidate_mesh_1 = cad_candidate->GetTransformedMeshPtr();
            pcl::PolygonMesh::Ptr candidate_mesh_2 = obj_it->second->GetTransformedMeshPtr();

            // Compute collision with penetration depth
            std::vector<Eigen::Vector3f> contact_locations;
            std::vector<float> penetration_depths;
            if (ComputeMeshMeshCollision(candidate_mesh_1, candidate_mesh_2,
                                         contact_locations, penetration_depths, true))
            {
                float max_penetration = std::abs(*max_element(penetration_depths.begin(),
                                                              penetration_depths.end()));
                total_penetration += max_penetration;
                collided_objects.push_back(collide_to_check.first);
            }
        }
    }
    return total_penetration;
}


void MapProcessingNode::FillContactGraph(const std::vector<Obj3D::Ptr>& objects,
        const std::unordered_map<int, OBBox>& gt_objects,
        std::unordered_map<Obj3D::Ptr, ObjCADCandidate::Ptr>& map_candidate,
        pgm::ParseGraphMap::Ptr& contact_graph)
{
    // object and its current supporting plane, transform of parent
    std::vector<std::pair<Obj3D::Ptr, std::pair<Eigen::Vector4f, Eigen::Matrix4f>>> obj_to_check;
    // Fill contact graph a top-down manner (spatially bottom-up)
    for (int i = 0; i < objects.size(); i++)
        // Only include nodes without supporting parent, e.g. Floor, Background
        if (objects[i]->GetSupportingParent().first == nullptr)
            obj_to_check.push_back({objects[i],
                                    {objects[i]->GetSupportingParent().second,
                                     Eigen::Matrix4f::Identity()}});

    while (obj_to_check.size() > 0)
    {
        Obj3D::Ptr current_object = obj_to_check[0].first;
        Eigen::Vector4f supported_plane = obj_to_check[0].second.first;
        Eigen::Matrix4f parent_transform = obj_to_check[0].second.second;
        pgm::ObjectNode::Ptr object_node = std::dynamic_pointer_cast<pgm::ObjectNode>
            (contact_graph->getNode(current_object->id));
        if (object_node == nullptr)
            ROS_ERROR("Invalid node id in contact graph!");

        auto obj_it = map_candidate.find(current_object);
        if (obj_it != map_candidate.end())
        {
            ObjCADCandidate::Ptr candidate = obj_it->second;
            ObjCAD::Ptr cad = candidate->GetCADPtr();
            Eigen::Vector3f transformed_dims = candidate->GetScale()
                * (canonical_base_transforms[candidate->GetPoseID()].topLeftCorner(3, 3)
                   * cad->GetDims()).cwiseAbs();

            // Adjust height according to the supported plane of parent
            candidate->SetHeight(-supported_plane(3) + transformed_dims(ground)/2);
            // Transform without/with scale
            // Output original unaligned CAD pose with no scale or
            // Output aligned CAD pose with no scale
            Eigen::Matrix4f transform_saving = candidate->GetTransform(false,
                    !save_original_cad_pose_);
            Eigen::Matrix4f transform = candidate->GetTransform();

            Eigen::Matrix4f relative_transform = parent_transform.inverse() * transform_saving;
            Eigen::Quaternionf quat(relative_transform.topLeftCorner<3, 3>());
            pgm::Point pos(relative_transform(0, 3),
                           relative_transform(1, 3),
                           relative_transform(2, 3));
            pgm::Quaternion quaternion(quat.x(), quat.y(), quat.z(), quat.w());

            std::cout << current_object->category_name << current_object->id << std::endl;
            object_node->setCadDataset(cad->cad_dataset);
            object_node->setCadID(cad->cad_id);
            object_node->setScale(candidate->GetScale(save_with_cad_scale_));
            object_node->setPose(pos, quaternion);

            if (gt_objects.size() > 0)
            {
                // Add matched ground-truth label and iou for CAD
                std::vector<std::pair<int, float>> label_iou_vec;
                if (match_ground_truth_)
                    label_iou_vec = {{current_object->id, 1.0f}};
                else if (!current_object->IsLayout())
                {
                    for (const auto& gt_pair : gt_objects)
                    {
                        float iou = ComputeGroundedBoxIOU3D(candidate->GetAlignedBox(),
                                                            gt_pair.second, ground);
                        if (iou > 0.0f)
                            label_iou_vec.push_back(std::make_pair(gt_pair.first, iou));
                    }
                    // Sort it
                    auto cmp = [](const auto & a, const auto & b) -> bool
                        {return a.second > b.second;};
                    std::sort(label_iou_vec.begin(), label_iou_vec.end(), cmp);
                }
                object_node->setIoUs(label_iou_vec);
            }

            for (const auto& child : current_object->GetSupportingChildren())
            {
                Obj3D::Ptr child_object = child.first;
                Eigen::Vector4f plane = cad->GetPlanes()
                    [candidate->GetSupportingPlaneMatch()[child.second]];
                Eigen::Vector4f transformed_supporting_plane =
                    TransformPlaneLocalToGlobal(plane, transform);

                obj_to_check.push_back({child_object,
                                        {transformed_supporting_plane,
                                         transform_saving}});
            }
        }
        else  // No CAD candidate
        {
            for (const auto& child : current_object->GetSupportingChildren())
            {
                Obj3D::Ptr child_object = child.first;
                Eigen::Vector4f plane = current_object->GetSupportingPlanes()[child.second].second;
                obj_to_check.push_back({child_object,
                                        {plane, Eigen::Matrix4f::Identity()}});
            }
        }
        obj_to_check.erase(obj_to_check.begin());
    }

    if (if_verbose_)
    {
        std::vector<pgm::NodeBase::Ptr> nodes = contact_graph->getNodes();
        std::cout << "PG Nodes" << std::endl;
        for (const auto& n : nodes)
            std::cout << n << std::endl;
    }
}


Eigen::Matrix4f MapProcessingNode::GetRefineTransform(const Eigen::Vector4f& x,
        const float object_ground_dim, const float cad_ground_dim)
{
    // x = [scale, d_theta, d_u, d_v], no scale
    Eigen::Matrix4f refine_transform = Eigen::Matrix4f::Zero();
    refine_transform(ground, ground) = 1.0;
    refine_transform(3, 3) = 1.0;
    refine_transform((ground+1)%3, (ground+1)%3) = cos(x(1));
    refine_transform((ground+1)%3, (ground+2)%3) = -sin(x(1));
    refine_transform((ground+2)%3, (ground+1)%3) = sin(x(1));
    refine_transform((ground+2)%3, (ground+2)%3) = cos(x(1));
    refine_transform((ground+1)%3, 3) = x(2);
    refine_transform((ground+2)%3, 3) = x(3);
    refine_transform(ground, 3) = x(0) * cad_ground_dim/2 - object_ground_dim/2;

    return refine_transform;
}


void MapProcessingNode::Run()
{
    //// Load CAD models
    ROS_INFO("Load CAD database...");
    std::unordered_map<std::string, std::vector<ObjCAD::Ptr>> cad_database;
    LoadObjectDatabase(cad_database);

    //// Load map object segments
    std::vector<Obj3D::Ptr> objects;
    // if (match_ground_truth_ && ground_truth_prefix_.size() > 0)
    // {
    //     ROS_INFO("Load ground truth panoptic segments...");
    //     LoadGroundTruthSegments(objects);
    // }
    // else
    // {
        ROS_INFO("Load panoptic segments...");
        LoadInstanceSegments(cad_database, objects);
    // }

    //// Load ground truth (if available)
    std::unordered_map<int, OBBox> gt_objects;
    // if (ground_truth_prefix_.size() > 0)
    //     LoadGroundTruthSegments(gt_objects);

    // Visualize scene mesh
    if (visualize_scene_mesh_)
    {
        std::vector<int> viz_ids;
        std::vector<pcl::PolygonMesh::Ptr> viz_meshes;
        std::vector<OBBox> viz_boxes;
        for (const auto& object : objects)
        {
            viz_meshes.push_back(object->GetMeshPtr());
            viz_boxes.push_back(object->GetBox());
            viz_ids.push_back(object->id);
        }
        viewer->AddPolygonMeshes(viz_meshes, viz_ids);
        viewer->AddCubes(viz_boxes);
        viewer->VisualizeOnce(viz_ids);
    }


    //// Build contact graph
    // Decide supporting parents for all objects
    ROS_INFO("Compute supporting relations...");
    std::unordered_map<Obj3D::Ptr,
                       std::unordered_map<Obj3D::Ptr, Eigen::Vector4f>> parent_child_map;
    // Nodes connected to the root node in the contact graph
    std::queue<Obj3D::Ptr> obj_to_check;
    DecideSupportingParents(objects, parent_child_map, obj_to_check);

    // Refine objects using the supporting planes
    UpdateObjectsViaSupportingRelations(objects, parent_child_map, obj_to_check);

    // Bottom-up build the contact graph
    ROS_INFO("Initialize contact graph...");
    pgm::ConceptNode root(0, "Room");
    pgm::ParseGraphMap::Ptr contact_graph (new pgm::ParseGraphMap(root));
    BuildContactGraph(parent_child_map, gt_objects, obj_to_check, contact_graph);
    if (contact_graph->getNodes().size() != objects.size()+1)
        ROS_ERROR("Missing object nodes in the contact graph!");

    if (save_contact_graph_)
    {
        std::string str_out = contact_graph->dump();
        ROS_INFO("Saving initialized contact graph!");
        std::string path = output_folder_ + "/contact_graph";
        makePath(path, 0777);
        contact_graph->save(path + "/contact_graph_seg.json");
    }

    if (match_cad_)
    {
        //// Coarse CAD matching
        // Scoring-based coarse CAD matching
        ROS_INFO("Match instance segments to CAD...");
        std::unordered_map<Obj3D::Ptr, std::vector<ObjCADCandidate::Ptr>> cad_candidates_map;
        MatchCADToSegmentsCoarse(objects, cad_database, cad_candidates_map);


        //// Validate physics and supporting affordance
        //// and return k best candidates for each map object
        ROS_INFO("Prune CAD candidates using supporting affordance...");
        std::unordered_map<Obj3D::Ptr, std::vector<ObjCADCandidate::Ptr>> cad_selected_candidates;
        for (auto& cad_candidates_pair : cad_candidates_map)
        {
            std::cout << cad_candidates_pair.first->category_name
                      << cad_candidates_pair.first->id << ": ";
            int count = 0;
            std::vector<ObjCADCandidate::Ptr> candidate_vec;
            for (auto& candidate : cad_candidates_pair.second)
            {
                // Check supported area & supporting affordance
                // If the 2D bbox area of those sampled CAD points close to CAD model bottom surface
                //      is greater than 0.3 * the 2D bbox area of CAD bbox,
                //      supporting area is sufficient.
                if (!ValidateSupportedArea(candidate, 0.3))
                    continue;
                else if (ValidateSupportingAffordance(candidate))
                {
                    candidate_vec.push_back(candidate);
                    count++;
                    std::cout << candidate->GetCADPtr()->cad_id << " ";
                }

                if (count > k_cad_candidates_)
                    break;
            }
            cad_selected_candidates.insert({cad_candidates_pair.first, candidate_vec});
            std::cout << std::endl;
        }


        //// Fine optimization-based CAD alignment and matching
        ROS_INFO("Compute potential collision pairs...");
        // Compute possible collisions between non-supporting objects
        std::unordered_map<Obj3D::Ptr, std::unordered_map<Obj3D::Ptr, float>> possible_collision_map;
        ComputePotentialCollisionPairs(cad_selected_candidates, possible_collision_map);
        ROS_INFO("Align CAD models...");
        MatchCADToSegmentsFine(objects, cad_selected_candidates);

        //// Global collision_free regulation
        ROS_INFO("Global physical violation check...");
        std::unordered_map<Obj3D::Ptr, ObjCADCandidate::Ptr> map_candidate;
        GlobalRegulation(objects, cad_selected_candidates, possible_collision_map, map_candidate);

        //// Fill in and save contact graph
        ROS_INFO("Update & save contact graph...");
        FillContactGraph(objects, gt_objects, map_candidate, contact_graph);
        if (save_contact_graph_)
        {
            std::string str_out = contact_graph->dump();
            ROS_INFO("Saving contact graph!");
            std::string path = output_folder_ + "/contact_graph";
            makePath(path, 0777);
            contact_graph->save(path + "/contact_graph_cad.json");
        }

        //// Visualize the scene
        if (visualize_final_scene_)
        {
            ROS_INFO("Visualize scene!");
            std::vector<OBBox> objects_viz;
            std::vector<pcl::PolygonMesh::Ptr> meshes;
            std::vector<int> id;
            for (int i = 0; i < objects.size(); i++)
            {
                auto obj_it = map_candidate.find(objects[i]);
                if (obj_it != map_candidate.end())
                {
                    ObjCADCandidate::Ptr candidate = obj_it->second;
                    pcl::PolygonMesh::Ptr mesh = candidate->GetTransformedMeshPtr();
                    OBBox box = candidate->GetAlignedBox();

                    meshes.push_back(mesh);
                    objects_viz.push_back(box);
                }
                else
                    meshes.push_back(objects[i]->GetMeshPtr());
                id.push_back(objects[i]->id);
            }
            viewer->AddPolygonMeshes(meshes, id);
            viewer->AddCubes(objects_viz);
            viewer->VisualizeOnce(id);
        }
    }

    ROS_INFO("Success!");
}

}  // namespace MapProcessing


int main(int argc, char **argv)
{
    ros::init(argc, argv, "map_processing");
    // ros::start();

    ros::NodeHandle node_handle("~");  // resolve namespace of node handle
    MapProcessing::MapProcessingNode map_processing_node(node_handle);
    map_processing_node.Run();

    // ros::spin(); // spin() will not return until the node has been shutdown

    // ros::shutdown();

    return 0;
}
