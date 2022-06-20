import numpy as np
import open3d as o3d
import math
import random

def is_nan(num):
    if num == num:
        return False
    else:
        return True


def str_to_np_array(str, delimiter, datatype):
    if not is_nan(str):
        str_list = str.split(delimiter)
        np_array = np.asarray(str_list, dtype=datatype)
    else:
        np_array = np.array([])
    return np_array


def np_array_to_str(array, delimiter):
    array_list = list(array)
    str_list = [str(x) for x in array_list]
    string = delimiter.join(str_list)
    return string


def np_to_str(matrix, delimiter):
    matrix_list = matrix.tolist()
    # print(matrix_list)
    str_list = [str(x) for lis in matrix_list for x in lis ]
    string = delimiter.join(str_list)
    return string


def detect_planes_from_mesh(mesh, distance_threshood, angle_threshood, max_iterations, min_area, max_plane_num):
    surface_normals = np.asarray(mesh.triangle_normals)
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # Represent surfaces as (num_triangle * 3 * 3)
    triangle_array = vertices[triangles.reshape(len(triangles)*3)].reshape(len(triangles),3,3)
    triangle_areas = get_triangle_area(triangle_array)
    triangle_centers = np.mean(triangle_array, axis=1)

    plane_list = []
    while(1):
        if len(triangle_array) == 0:
            print('[ WARNING ] Exiting extract planes: no triangles left')
            break
        else:
            print(f'Iter {len(plane_list)}/{max_plane_num}: num_triangles={len(triangle_array)}')
        plane, inliers = estimate_plane_ransac_from_mesh(triangle_array, surface_normals, triangle_centers, triangle_areas,
                                    max_iterations=max_iterations, min_area=min_area, distance_threshood=distance_threshood, angle_threshood=angle_threshood)
        if len(inliers) == 0:
            print('Exiting extract planes: no inliers')
            break
        else:
            # Update outliers
            outliers = np.logical_not(inliers)
            # outliers = np.arange(len(triangle_array))[outliers]
            triangle_array = triangle_array[outliers]
            surface_normals = surface_normals[outliers]
            triangle_centers = triangle_centers[outliers]
            triangle_areas = triangle_areas[outliers]

            plane_list.append(plane)
            if len(plane_list) >= max_plane_num:
                print('Exiting extract planes: max_plane_num reached')
                break

    return plane_list


def get_triangle_area(triangle_array):
    edge_1 = triangle_array[:,1,:] - triangle_array[:,0,:]
    edge_2 = triangle_array[:,2,:] - triangle_array[:,0,:]
    triangle_area = 0.5 * np.linalg.norm(np.cross(edge_1, edge_2, axis=1),axis=1)
    triangle_area = triangle_area.reshape(len(triangle_area),1)

    return triangle_area


def estimate_plane_ransac_from_mesh(triangle_array, surface_normals, triangle_centers, triangle_areas, min_area,
                                            max_iterations, distance_threshood, angle_threshood):
    '''
    triangle_array: three vertices of each triangle, (num_triangle, 3, 3) array
    surface_normals: triangle normals, (num_triangle, 3) array, MUST be normalized
    triangle_centers: centers of triangles, (num_triangle, 3) array
    triangle_areas: areas of triangles, (num_triangle, 1) array
    max_iterations: maximum iterations of ransac sampling
    min_area: minimum area for a valid plane
    distance_threshood, angle_threshood: tolerance for distance(m)/angle(degree) error
    return: plane model [a,b,c,d] array
            inliers [True, False, ...] (1, num_triangle) array
    '''
    triangle_size = len(triangle_array)
    best_inlier_area = 0
    best_inliers = []
    # best_removal = None
    best_plane = []
    random.seed(12345)

    for i in range(max_iterations):
        # Sampe surface
        idx = random.sample(range(triangle_size), 1)
        # Compute initial plane parameters
        plane_normal = np.squeeze(surface_normals[idx,:])
        triangle_center = np.squeeze(triangle_centers[idx,:])
        d = -triangle_center.dot(plane_normal)

        # Check inliers
        inlier_1 = surface_normals.dot(plane_normal) > np.ones(triangle_size) * math.cos(angle_threshood*math.pi/180)
        inlier_2 = np.absolute(triangle_centers.dot(plane_normal) + np.ones(triangle_size)*d) < np.ones(triangle_size) * distance_threshood
        inliers = np.logical_and(inlier_1, inlier_2).T.reshape(triangle_size)
        inlier_area = np.sum(triangle_areas[inliers])

        # Compute min_area corresponds to plane normal
        # absolute_normal = np.absolute(plane_normal)
        # max_index = np.argsort(absolute_normal)[-1]
        # if max_index == 0:
        #     cos_angle = absolute_normal.dot(np.array([1,0,0]))
        #     min_area = aligned_dims[1]*aligned_dims[2]/cos_angle/3
        # elif max_index == 1:
        #     cos_angle = absolute_normal.dot(np.array([0,1,0]))
        #     min_area = aligned_dims[0]*aligned_dims[2]/cos_angle/3
        # else:
        #     cos_angle = absolute_normal.dot(np.array([0,0,1]))
        #     min_area = aligned_dims[0]*aligned_dims[1]/cos_angle/3

        # if

        if inlier_area > best_inlier_area and inlier_area > min_area:
            best_inlier_area = inlier_area
            best_inliers = inliers
            best_plane = np.array([plane_normal[0],plane_normal[1],plane_normal[2],d])

            if i > max_iterations/4:
                print(f'RANSAC Premature stop: max_iterations/4 reached {i} > {max_iterations//4}')
                break

    with np.printoptions(precision=3, suppress=True):
        if len(best_inliers) == 0:
            assert inlier_area <= min_area, f'inlier_area {inlier_area} greater than min_area {min_area} but not found'
            assert i == max_iterations-1, f'max_iterations not reached but not found'

            print(f'RANSAC fail to find plane: inlier_area <= min_area. '
                  f'{inlier_area} <= {min_area}')
        else:
            print(f'RANSAC result: best_plane={best_plane}, num_inlier_trigs={np.count_nonzero(best_inliers)}, inlier_area={best_inlier_area}')

    return best_plane, best_inliers



def detect_planes_from_pc(point_cloud, threshood, max_plane_num, viz=False):
    '''
    point_cloud: open3d point cloud with normals
    up_axis: direction of up-axis
    threshood: threshood for minimal points of a plane
    return: list of numpy arrays
    '''
    compute_normal_flag = True
    plane_list = []
    inlier_cloud_list = []
    outlier_cloud = point_cloud
    pc_size = len(point_cloud.points)
    distance_threshood = np.amin(point_cloud.get_max_bound() - point_cloud.get_min_bound())*0.05
    if distance_threshood > 0.01:
        distance_threshood = 0.01
    while (1):
        # plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=distance_threshood,
        #                                  ransac_n=5,
        #                                  num_iterations=500)
        plane_array, inliers = estimate_plane_ransac(outlier_cloud, ransac_n=3, max_iterations=600, min_inliers=threshood * pc_size,
                                                            distance_threshood=distance_threshood, angle_threshood=8)

        if len(inliers) == 0:
            # if len(plane_list) == 0 and compute_normal_flag:
            #     outlier_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            #     compute_normal_flag = False
            # else:
            break
        else:
            inliers = list(inliers)
            inlier_cloud = outlier_cloud.select_by_index(inliers)
            outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
            # point_normals = np.asarray(inlier_cloud.normals)
            # inlier_cloud_normals_mean = np.mean(point_normals, axis=0)
            # plane_array = np.asarray(plane_model)
            # if inlier_cloud_normals_mean.dot(plane_array[0:3]) < 0:
            #     plane_array = -plane_array

            # print(get_plane_normal_error(plane_array[0:3], point_normals))
            # if get_plane_normal_error(plane_array[0:3], point_normals) > 1.0:
            #     continue

            color = list(np.random.random_sample(3))
            inlier_cloud.paint_uniform_color(color)
            inlier_cloud_list.append(inlier_cloud)
            # o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud])
            plane_list.append(plane_array)
            if len(outlier_cloud.points) < threshood * pc_size or len(plane_list) > max_plane_num:
                break
    if viz:
        o3d.visualization.draw_geometries(inlier_cloud_list + [outlier_cloud])
    return plane_list


def estimate_plane_ransac(point_cloud, ransac_n, max_iterations, min_inliers, distance_threshood, angle_threshood):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    # Nomalize point normals
    normals = np.divide(normals, np.linalg.norm(normals,axis=1).reshape(len(normals),1))
    # Augment points into [x, y, z, 1]
    augment_points = np.ones((len(points), 4))
    augment_points[:,0:3] = points

    best_inlier_num = 0
    best_inliers = None
    best_removal = None
    best_plane = None
    best_plane_normal = None

    for i in range(max_iterations):
        random.seed()
        # Sampe points
        idx = random.sample(range(len(points)), ransac_n)
        init_aug_points = augment_points[idx, :]

        # Estimate plane coefficients using svd
        init_plane = np.linalg.svd(init_aug_points)[-1][-1, :].T
        init_plane_normal = init_plane[0:3]
        init_plane_normal = init_plane_normal/np.linalg.norm(init_plane_normal)

        # Inliners considering distance constraints
        inlier_1 = np.absolute(augment_points.dot(init_plane)) < np.ones(len(points)) * distance_threshood

        # Inliners considering angle constraints
        normals_mul_plane_normal = normals.dot(init_plane_normal)
        inlier_2_1 =  normals_mul_plane_normal > np.ones(len(points)) * math.cos(angle_threshood*math.pi/180)
        inlier_2_2 = -normals_mul_plane_normal > np.ones(len(points)) * math.cos(angle_threshood*math.pi/180)

        # Check distance and angle error
        inliers1 = np.logical_and(inlier_1, inlier_2_1)
        inliers2 = np.logical_and(inlier_1, inlier_2_2)
        inlier1_sum = np.sum(inliers1)
        inlier2_sum = np.sum(inliers2)
        if inlier1_sum > inlier2_sum:
            inliers = inliers1
            inlier_num = inlier1_sum
        else:
            inliers = inliers2
            inlier_num = inlier2_sum
            init_plane_normal = -init_plane_normal

        if inlier_num > best_inlier_num:
            best_inlier_num = inlier_num
            best_inliers = inliers.T.reshape(len(inliers))
            # Set all points within distance threshood as removal
            best_removal = inlier_1.T.reshape(len(inlier_1))
            # best_removal = inliers.T.reshape(len(inliers))

            best_plane_normal = init_plane_normal
            if best_inlier_num > min_inliers and i > max_iterations/2:
                break

    if best_inlier_num > min_inliers:
        # Recompute best plane
        inlier_aug_points = augment_points[best_inliers]
        best_plane = np.linalg.svd(inlier_aug_points)[-1][-1, :].T
        if best_plane[0:3].dot(best_plane_normal) < 0:
            best_plane = -best_plane

        # best_inliers = np.arange(len(points))[best_inliers]
        best_removal = np.arange(len(points))[best_removal]
    else:
        best_plane = []
        best_removal = []

    return best_plane, best_removal


def get_plane_normal_error(plane_normal, point_normals):
    error_array = 1-np.divide(point_normals, np.linalg.norm(point_normals,axis=1)).dot(plane_normal/np.linalg.norm(plane_normal))
    error = error_array.dot(error_array)/len(point_normals)
    return error


def transform_planes(plane_list, T):
    invT = np.linalg.inv(T)
    plane_list_transformed = []
    for plane in plane_list:
        plane_transformed = invT.dot(plane)
        plane_list_transformed.append(plane_transformed)
        # print(plane_transformed.tolist())
    return plane_list_transformed


def get_visible_points(point_cloud, camera_list):
    # visible_indices_set = Set([])
    visible_indices = []
    diameter = np.linalg.norm(np.asarray(point_cloud.get_max_bound()) - np.asarray(point_cloud.get_min_bound()))
    radius = diameter * 100
    for camera_pos in camera_list:
        camera = np.array(camera_pos) * diameter
        _, pt_map = point_cloud.hidden_point_removal(camera, radius)
        # visible_indices_set.add(tuple(pt_map))
        visible_indices = visible_indices + pt_map
    point_cloud_visible = point_cloud.select_by_index(visible_indices)

    return point_cloud_visible


def draw_planes(plane_list, size):
    line_set_list = []
    color_list = []
    for plane in plane_list:
        a = plane[0]
        b = plane[1]
        c = plane[2]
        d = plane[3]
        nn = a**2+b**2+c**2
        p0 = np.array([-a*d/nn, -b*d/nn, -c*d/nn])  # drop feet
        n = plane[0:3]/nn**0.5  # plane normal

        min_index = np.argmin(np.absolute(plane[0:3]))
        if min_index == 0:
            align_vec = np.array([1,0,0])
        elif min_index == 1:
            align_vec = np.array([0,1,0])
        else:
            align_vec = np.array([0,0,1])

        u = align_vec - n.dot(align_vec)*n
        u = u/np.linalg.norm(u)
        v = np.cross(n, u)

        p1 = p0 + size * u + size * v
        p2 = p0 + size * u - size * v
        p3 = p0 - size * u + size * v
        p4 = p0 - size * u - size * v
        points = [list(p1), list(p2), list(p3), list(p4)]

        lines = [[0, 1], [0, 2], [1, 3], [2, 3]]
        color = list(np.random.random_sample(3))
        color_list.append(color)
        colors = [color for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines),)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set_list.append(line_set)
    return line_set_list, color_list


def draw_plane_normal(plane_list, colors, scale=0.01):
    arrow_list = []
    i = 0
    for plane in plane_list:
        mesh= o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=1.0*scale, cone_radius=1.5*scale, cylinder_height=5.0*scale, cone_height=4.0*scale)
        vector = plane[0:3]
        vector = vector/np.linalg.norm(vector)
        T = np.eye(4)
        T[0:3,2] = vector
        random_vec = np.random.random_sample(3)
        random_vec = random_vec - vector.dot(random_vec)*vector
        random_vec = random_vec/np.linalg.norm(random_vec)
        T[0:3,0] = random_vec
        T[0:3,1] = np.cross(T[0:3,2],T[0:3,0])
        nn = plane[0]**2 + plane[1]**2 + plane[2]**2
        T[0,3] = -plane[0]*plane[3]/nn
        T[1,3] = -plane[1]*plane[3]/nn
        T[2,3] = -plane[2]*plane[3]/nn
        mesh = mesh.transform(T)
        mesh.paint_uniform_color(colors[i])
        i = i+1
        arrow_list.append(mesh)
    return arrow_list
