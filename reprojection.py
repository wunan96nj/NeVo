import numpy as np
import imageio
import math
import cv2
import open3d as o3d
def rodrigues_rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def load_data(frame_index):
    """ Load the saved data for a given frame index """
    base_path = "saved_ray_value_ijk/frame_{}_".format(frame_index)
    ray_pts = np.loadtxt(base_path + "ray_pts.txt", delimiter=",")
    value_at_point = np.loadtxt(base_path + "value_at_point.txt", delimiter=",")

    # weights = np.fromfile(weights_list_filename, dtype=np.float32)
    # RGB = np.fromfile(raw_rgb_list_filename, dtype=np.float32).reshape((-1, 3))  # Assuming RGB is a 2D array
    # ray_id = np.fromfile(ray_id_list_filename, dtype=np.int64)


    weights = np.fromfile(base_path + "weights.bin", dtype=np.float32)
    raw_rgb = np.fromfile(base_path + "raw_rgb.bin", dtype=np.float32).reshape(-1, 3)  # Assuming 3 channels for RGB
    return ray_pts, value_at_point, weights, raw_rgb

def reproject_and_accumulate(ray_pts, weights, raw_rgb, intrinsic, extrinsic, image_size):
    """ Reproject points to an image and accumulate colors """
    weighted_rgb = raw_rgb#weights[:, None] * raw_rgb  # Apply weights to colors

    # Transform 3D points to the camera coordinate system
    ones = np.ones((ray_pts.shape[0], 1))
    homogenous_ray_pts = np.hstack((ray_pts, ones))
    camera_coords = extrinsic.dot(homogenous_ray_pts.T).T[:, :3]

    # Project points onto the image plane
    projected_pts = intrinsic.dot(camera_coords.T).T
    projected_pts /= projected_pts[:, 2][:, None]  # Normalize by the depth

    # Initialize the image
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.float32)

    # Accumulate colors onto the image
    for pt, color in zip(projected_pts, weighted_rgb):
        x, y = round(pt[0]), round(pt[1])
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            if np.all(image[y, x] == 0):
                image[y, x] += color
            else:
                image[y, x] = np.minimum(image[y, x], color)
                #image[y, x] += color


    # Set untouched pixels to white
    untouched_mask = np.all(image == 0.0, axis=-1)  # Find pixels that are still black
    #image[untouched_mask] = [1.0, 1.0, 1.0]  # Set these pixels to white

    return image

def reproject_and_accumulate_pts(ray_pts, raw_rgb, intrinsic, extrinsic, image_size):
    """ Reproject points to an image and accumulate colors """
    weighted_rgb = raw_rgb  # Apply weights to colors

    # Transform 3D points to the camera coordinate system
    ones = np.ones((ray_pts.shape[0], 1))
    homogenous_ray_pts = np.hstack((ray_pts, ones))
    camera_coords = extrinsic.dot(homogenous_ray_pts.T).T[:, :3]

    # Project points onto the image plane
    projected_pts = intrinsic.dot(camera_coords.T).T
    projected_pts /= projected_pts[:, 2][:, None]  # Normalize by the depth
    print(projected_pts)

    # Initialize the image
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.float32)

    failed_pixels = []
    # Accumulate colors onto the image
    for pt, color in zip(projected_pts, weighted_rgb):
        x, y = round(pt[0]), round(pt[1])
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            if np.all(image[y, x] == 0):
                image[y, x] += color


    # Function to get the color from surrounding pixels
    def get_color_from_neighbors(x, y, image):
        for dx in [-1,]:
            for dy in [-1]:
                # Skip the center pixel
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < image_size[0] and 0 <= ny < image_size[1]:
                    if not np.all(image[ny, nx] == 0):
                        return image[ny, nx]
        return None
    pixels_to_update = []
    # Loop through all pixels in the image
    for y in range(image_size[1]):
        for x in range(image_size[0]):
            if np.all(image[y, x] == 0):
                color = get_color_from_neighbors(x, y, image)
                if color is not None:
                    pixels_to_update.append((x, y, color))
    # Combine with the original color
    for x, y, color in pixels_to_update:
        # Combine the color with the original color at (x, y)
        # Assuming a simple replacement, but you can modify this logic as needed
        image[y, x] = color

    # Set untouched pixels to white
    untouched_mask = np.all(image == 0.0, axis=-1)  # Find pixels that are still black
    #image[untouched_mask] = [1.0, 1.0, 1.0]  # Set these pixels to white

    return image


def create_point_cloud_from_depth(rgb_image_path, depth_data_path, intrinsic_matrix, extrinsic_matrix):
    # Load the RGB image
    rgb_image = cv2.imread(rgb_image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Load the depth data and remove the redundant dimension
    depth_data = np.load(depth_data_path)
    depth_data = np.squeeze(depth_data)  # This removes the singleton dimension
    print(depth_data.min())
    min_depth = depth_data.min()

    # Get the width and height from the depth data
    height, width = depth_data.shape

 

    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
    # Generate points from the depth data
    points = []
    colors = []
    for v in range(height):
        for u in range(width):
            depth = depth_data[v, u]
            if depth == min_depth:  # Skip zero depth values
                continue

            # Get the color at this pixel
            color = rgb_image[v, u]



            # # Check if the color is white (you can adjust the threshold as needed)
            # if np.all(color == [255, 255, 255]):
            #     continue  # Skip white points
            # Threshold for near-white color
            near_white_threshold = 230

            # Check if the color is nearly white
            if np.all(color >= near_white_threshold):
                continue  # Skip nearly white points

            
            # Compute the 3D point in camera coordinates
            x = (u - intrinsic_matrix[0, 2]) * depth / intrinsic_matrix[0, 0]
            y = (v - intrinsic_matrix[1, 2]) * depth / intrinsic_matrix[1, 1]
            z = depth

            # Create a 4x1 homogeneous coordinate matrix for the 3D point
            point_homogeneous = np.array([x, y, z, 1])

            # Apply the extrinsic matrix (rotation and translation)
            transformed_point = extrinsic_matrix @ point_homogeneous

            # Append the transformed point to the points list
            points.append(transformed_point[:3])  # Only take x, y, z

            # Append corresponding color
            colors.append(rgb_image[v, u] / 255.0)

            # # Append the point to the points list
            # points.append([x, y, z])

            # # Append corresponding color
            # colors.append(rgb_image[v, u] / 255.0)

    #Create a point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    #Save the point cloud
    o3d.io.write_point_cloud("output_point_cloud.ply", pcd)

    print("Point cloud saved as 'output_point_cloud.ply'")

    return np.array(points), np.array(colors)


for id in range(3000):
    # Example usage
    frame_index = 0+id
    ray_pts, _, weights, raw_rgb = load_data(frame_index)
    print(ray_pts.shape)

    # Define your intrinsic and extrinsic matrices and image size
    intrinsic = np.array([[4176.1, 0.0, 960],
                                [0.0, 4176.1, 540],
                                [0.0, 0.0, 1.0]])

    # Construct camera matrix
    nR = np.array([xaxis, yaxis, lookat, pos]).T
    nR = np.concatenate([nR, np.array([[0, 0, 0, 1]])], axis=0)



    # Invert nR to get the extrinsic matrix
    extrinsic = np.linalg.inv(nR)
    image_size = (1920, 1080)  # Replace with actual image dimensions


    # Replace these with your actual file paths
    rgb_image_path = "output/soldier/render_360_rerf_252/{:03d}.png".format(cam_id)
    depth_data_path = "output/soldier/render_360_rerf_252/{:03d}_original_depth.npy".format(cam_id)
    print(rgb_image_path, depth_data_path)

    points, colors = create_point_cloud_from_depth(rgb_image_path, depth_data_path, intrinsic, extrinsic)


    # Construct camera matrix
    nR_new = np.array([xaxis, yaxis, lookat, pos]).T
    nR_new = np.concatenate([nR_new, np.array([[0, 0, 0, 1]])], axis=0)

    # Invert nR to get the extrinsic matrix
    extrinsic_new = np.linalg.inv(nR_new)
    image_size = (1920, 1080)  # Replace with actual image dimensions
    # Reproject and accumulate
    reprojected_image = reproject_and_accumulate_pts(points, colors, intrinsic, extrinsic_new, image_size)


    reprojected_image_copy = np.copy(reprojected_image)
    untouched_mask = np.all(reprojected_image_copy == 0.0, axis=-1)  # Find pixels that are still black
    reprojected_image_copy[untouched_mask] = [1.0, 1.0, 1.0]  # Set these pixels to white
    # Optionally, normalize and save the image
    #reprojected_image = np.clip(reprojected_image / reprojected_image.max(), 0, 1)
    # Convert floating point image in the range [0, 1] to 8-bit image [0, 255]
    def to8b(image):
        return (255 * np.clip(image, 0, 1)).astype(np.uint8)

    # Convert to 8-bit and save
    rgb8 = to8b(reprojected_image_copy)
    imageio.imwrite('saved_reprojection_image/{:03d}_test_reprojection_all.jpg'.format(id), rgb8)

    # print(ray_pts.shape)

    # Reproject and accumulate
    reprojected_image_intermediate = reproject_and_accumulate(ray_pts, weights, raw_rgb, intrinsic, extrinsic, image_size)

    # Optionally, normalize and save the image
    #reprojected_image = np.clip(reprojected_image / reprojected_image.max(), 0, 1)
    # Convert floating point image in the range [0, 1] to 8-bit image [0, 255]
    def to8b(image):
        return (255 * np.clip(image, 0, 1)).astype(np.uint8)

    # Convert to 8-bit and save
    rgb8 = to8b(reprojected_image_intermediate)
    imageio.imwrite('saved_reprojection_image/{:03d}_test_reprojection.jpg'.format(id), rgb8)