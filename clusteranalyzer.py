# -*- coding: utf-8 -*-
""" Generates an analysis of video data.

@author Cormak Weeks and Amory Gaylord
"""

import cv2
import numpy as np
import os
import pandas as pd
import csv  
import glob
import json

class analyzer():
    ''' Analyzes a set of contours from a video, including generating vector fields and shape
    information.

    Attributes
    ----------
    labels : str
        The .json file containing the video labels in COCO format
    img_directory : str 
        The path to the directory containing the frames from a video
    dataset_name : str
        The name of the generated dataset (e.g. name of the video)
    frames_df_csv : str
        The path to the csv of frame information
    frames_df : DataFrame
        A DataFrame containing information about frames
    contours_df : DataFrame
        A DataFrame containing information about the contours

    '''

    REFERENCE_SHAPES = {
        "circle": cv2.circle(np.zeros((100, 100), dtype=np.uint8), (50, 50), 40, 255, -1),
        "triangle": np.array([[[10, 90], [50, 10], [90, 90]]], dtype=np.int32),
        "square": np.array([[[10, 10], [90, 10], [90, 90], [10, 90]]], dtype=np.int32),
        "diamond": np.array([[[50, 10], [90, 50], [50, 90], [10, 50]]], dtype=np.int32),
        "trapezoid": np.array([[[10, 90], [30, 10], [70, 10], [90, 90]]], dtype=np.int32),
        "pentagon": np.array([[[50, 10], [90, 30], [75, 80], [25, 80], [10, 30]]], dtype=np.int32),
        "hexagon": np.array([[[50, 5], [90, 30], [90, 70], [50, 95], [10, 70], [10, 30]]], dtype=np.int32),
        "heptagon": np.array([[[50, 5], [90, 25], [90, 75], [60, 95], [30, 95], [10, 75], [10, 25]]], dtype=np.int32),
        "octagon": np.array([[[40, 0], [60, 0], [90, 30], [90, 70], [60, 100], [40, 100], [10, 70], [10, 30]]], dtype=np.int32)
    }

    def __init__(self, labels, img_directory, dataset_name):
        self.labels = labels # json file
        self.img_directory = img_directory # directory of images
        self.dataset_name = dataset_name
        self.frames_df_csv = ""
        self.frames_df = self.generate_frames_df()
        self.contours_df = pd.read_json(self.labels, orient='records', columns=["segmentation", "area", "iscrowd", "image_id", "bbox", "category_id", "id", "ignore"])

        # TODO: MODIFY TRAINER_GUI SO COCO LABELS AREN'T DELETED

    def generate_frames_df(self):
        ''' Generates a DataFrame from the analyzer's image directory of video frames
        
        Parameters
        ----------
        None.

        Returns
        -------
        frames_df : DataFrame
            A DataFrame containing information about the video's frames

        '''
        
        # get list of file names
        files = sorted(glob.glob(self.img_directory + '*.txt'))

        # create iteration variables
        frame_list = []
        frame_number = 0

        # iterate through files
        for f in files:
            img = cv2.imread(f)
            
            # create a dict of frames for the read file
            frame_info = {
                "file_name": f,
                "frame_number": frame_number,
                "frame_width": img.shape[1],
                "frame_height": img.shape[0],
                "frame_channels": img.shape[2],
            }

            # append this frame's information
            frame_list.append(frame_info)

            # increment the counter
            frame_number += 1

        # convert the dict to a DataFrame
        frames_df = pd.DataFrame(frame_list)

        # manage directories
        if not os.path.exists("model/"):
            os.mkdir("model")
        if not os.path.exists("model/datasets/"):
            os.mkdir("model/datasets/")

        # save frames DataFrame to a csv 
        self.frames_df_csv = f"model/datasets/{self.dataset_name}_frames.csv"
        frames_df.to_csv(self.frames_df_csv, index=False)

        # return the DataFrame
        return frames_df

    def merge_close_contours_within_frames(self, max_centroid_distance=200, max_edge_distance=5):
        ''' Merges contours within a given frame that are within a certain distance of each other

        Parameters
        ----------
        max_centroid_distance : int, optional
            The maximum number of pixels between 2 centroids for them to be merged.
            The default value is 200.
        max_edge_distance : int, optional
            The maximum number of pixels between 2 edges for them to be merged.
            The default value is 5.

        Returns
        -------
        merged_contours_list : list
            A list of updated contours that have been merged together.
        
        '''
        
        # set up new list
        merged_contours_list = []
        # convert existing contours DataFrame to a list
        # contours_list = self.contours_df.values.tolist()

        # get list of unique frames
        frames = self.contours_df["image_id"].unique().tolist()
        
        # set up iteration variable
        counter = 0

        # iterate through frames
        while counter <= frames.max():
            # skip if there was no detected contours in this frame
            if counter not in frames:
                merged_contours_list.append([])
                continue

            # the clusters already iterated through
            merged_clusters = []

            # get a DF containing only the contours of this frame
            frame_contours = self.contours_df[self.contours_df["image_id"] == counter]

            # iterate through contours in this frame
            for contour in frame_contours:
                # set up bools
                merged_with_existing = False
                merged_cluster = None

                # merged_clusters being the clusters already iterated through
                # iterate through the previously merged contours (worst case = n)
                for cluster in merged_clusters:
                    # iterate through the contours in the current merged cluster
                    for existing_contour in cluster:
                        # Calculate the centroid distance between the two contours
                        centroid_distance = np.linalg.norm(np.mean(contour, axis=0) - np.mean(existing_contour, axis=0))

                        # determine whether to centroid distance criteria is met
                        if centroid_distance <= max_centroid_distance:

                            # Compare the edges points of the two contours
                            # the current contour
                            for pt_contour in contour:
                                # the other contour
                                for pt_existing in existing_contour:
                                    # calculate distance
                                    edge_distance = np.linalg.norm(pt_contour - pt_existing)

                                    # once a distance less than the threshold is found, exit out of this loop 
                                    if edge_distance <= max_edge_distance:
                                        # add new contour to merged_cluster
                                        if merged_cluster is None:
                                            # Create a new merged cluster and add both contours
                                            merged_cluster = np.concatenate((contour, existing_contour))
                                            merged_with_existing = True
                                            break
                                        # merge this cluster with the previous cluster
                                        else:
                                            # Merge the contour into the existing merged cluster
                                            merged_cluster = np.concatenate((merged_cluster, contour))
                                            merged_with_existing = True
                                            break

                                if merged_with_existing:
                                    break

                        if merged_with_existing:
                            break

                    if merged_with_existing:
                        break

                if not merged_with_existing:
                    merged_clusters.append([contour])

                if merged_cluster is not None:
                    # Add the merged cluster to the list if it was formed
                    merged_clusters.append([merged_cluster])

            merged_contours_list.append(merged_clusters)

        return merged_contours_list

    def save_merged_contours(merged_contours_list, frames_df, dataset_name):
        contours_folder = "contours"
        merged_folder = "merged"
        dataset_folder = os.path.join("/content/drive/My Drive", "datasets", dataset_name)
        contours_folder_path = os.path.join(dataset_folder, contours_folder)
        merged_folder_path = os.path.join(contours_folder_path, merged_folder)

        if not os.path.exists(merged_folder_path):
            os.makedirs(merged_folder_path)

        for index, merged_clusters in enumerate(merged_contours_list):
            frame_name = frames_df.loc[index, "file_name"]
            merged_frame_path = os.path.join(merged_folder_path, frame_name)

            # Get frame dimensions from frames_df
            height = frames_df.loc[index, "frame_height"]
            width = frames_df.loc[index, "frame_width"]

            frame_with_merged_contours = np.zeros((height, width, 3), dtype=np.uint8)

            # Process each merged cluster separately
            for cluster in merged_clusters:
                # Convert cluster to a list of contours
                cluster_contours = [contour for contour in cluster]

                # Calculate convex hulls of individual contours
                convex_hulls = [cv2.convexHull(contour) for contour in cluster_contours]

                # Merge convex hulls to create a single convex hull for the cluster
                merged_convex_hull = np.vstack(convex_hulls)

                # Draw the merged convex hull directly onto the frame using fillPoly
                cv2.fillPoly(frame_with_merged_contours, [merged_convex_hull], (255, 255, 100))

            cv2.imwrite(merged_frame_path, frame_with_merged_contours)
    #---------------------------------------------------------------------------
    def generate_features(contours_list):
        features_list = []

        for frame_idx, contours in enumerate(contours_list):
            frame_features = []

            for contour in contours:
                # Calculate contour features
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                eccentricity = np.sqrt((M["m20"] - M["m02"]) ** 2 + 4 * M["m11"] ** 2) / (M["m20"] + M["m02"])
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area

                frame_features.append((cX, cY, area, perimeter, eccentricity, solidity))

            features_list.append(frame_features)

        return features_list
    #---------------------------------------------------------------------------
    def contour_rotation(contours_list):
        rotation_list = []

        for frame_idx in range(len(contours_list) - 1):
            rotation_frame = []

            for contour_idx, curr_contour in enumerate(contours_list[frame_idx]):
                next_frame_contours = contours_list[frame_idx + 1]
                if contour_idx >= len(next_frame_contours):
                    continue

                _, _, angle_curr = cv2.minAreaRect(curr_contour)
                _, _, angle_next = cv2.minAreaRect(next_frame_contours[contour_idx])
                rotation_angle = angle_next - angle_curr

                rotation_frame.append(rotation_angle)

            rotation_list.append(rotation_frame)

        return rotation_list
    #------------------------------------------------------------------------------
    def new_generate_contours(frames_df, dataset_name, epsilon=1):
        contours_folder = "merged_contours"
        dataset_folder = os.path.join("/content/drive/My Drive", "datasets", dataset_name)
        contours_folder_path = os.path.join(dataset_folder, contours_folder)
        if not os.path.exists(contours_folder_path):
            os.makedirs(contours_folder_path)

        new_contours_list = []  # Initialize list to store contours for each frame

        for index, row in frames_df.iterrows():
            frame_name = row["file_name"]
            frame_path = os.path.join(dataset_folder, "contours","merged", frame_name)
            frame = cv2.imread(frame_path)

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply thresholding or other preprocessing as needed
            _, threshold_frame = cv2.threshold(gray_frame, 25, 200, cv2.THRESH_BINARY)

            # Find contours in the thresholded frame
            contours, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter and store valid contours for this frame
            valid_contours = []
            for contour in contours:
                if cv2.contourArea(contour) >= 100:
                    # Simplify the contour using Douglas-Peucker algorithm
                    approx_contour = cv2.approxPolyDP(contour, epsilon, closed=True)
                    valid_contours.append(approx_contour)

            new_contours_list.append(valid_contours)

            # Create a copy of the frame to draw contours
            frame_with_contours = frame.copy()
            for contour in valid_contours:
                cv2.drawContours(frame_with_contours, [contour], 0, (0, 255, 100), 2)

            contours_frame_path = os.path.join(contours_folder_path, frame_name)
            cv2.imwrite(contours_frame_path, frame_with_contours)

        return new_contours_list
    #--------Functions---------------------
    generate_bounding_boxes(frames_info, dataset_name)
    generate_contours(frames_info, dataset_name, epsilon=5)
    contours_list = generate_contours(frames_info, dataset_name)
    features_list = generate_features(contours_list)
    merged_contours_list = merge_close_contours_within_frames(contours_list, max_centroid_distance=200, max_edge_distance=7)
    save_merged_contours(merged_contours_list, frames_info, dataset_name)
    rotation_list = contour_rotation(contours_list)
    new_contours_list=new_generate_contours(frames_info, dataset_name, epsilon=1)

    def merge_contours_on_frames(contours_list, new_contours_list, dataset_name):
        merged_contours_folder = "merged_contours"
        dataset_folder = os.path.join("/content/drive/My Drive", "datasets", dataset_name)
        train_folder = os.path.join(dataset_folder, "train")
        merged_contours_folder_path = os.path.join(dataset_folder, merged_contours_folder)

        if not os.path.exists(merged_contours_folder_path):
            os.makedirs(merged_contours_folder_path)

        for index, row in frames_info.iterrows():
            frame_name = row["file_name"]
            frame_path = os.path.join(train_folder, frame_name)
            frame = cv2.imread(frame_path)

            if frame is None:
                print("Error reading frame:", frame_name)
                continue

            # Draw contours from contours_list
            for contour in contours_list[index]:
                cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)

            # Draw contours from new_contours_list
            for contour in new_contours_list[index]:
                cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)

            merged_frame_path = os.path.join(merged_contours_folder_path, frame_name)
            cv2.imwrite(merged_frame_path, frame)

    # Assuming contours_list and new_contours_list are generated already
    merge_contours_on_frames(contours_list, new_contours_list, dataset_name)

    #Extracting Parameters

    def track_clusters(contours_list):
        tracked_positions = []  # List to store tracked positions of clusters

        # Initialize an empty dictionary to store centroids for each cluster ID
        centroids_dict = {}

        for frame_idx, contours in enumerate(contours_list):
            frame_positions = []

            for contour in contours:
                M = cv2.moments(contour)  # Calculate moments for the contour
                if M["m00"] == 0:
                    continue  # Skip if the contour has zero area

                cX = int(M["m10"] / M["m00"])  # Calculate x-coordinate of centroid
                cY = int(M["m01"] / M["m00"])  # Calculate y-coordinate of centroid

                if len(centroids_dict) == 0:
                    cluster_id = 0
                    centroids_dict[cluster_id] = (cX, cY)
                else:
                    cluster_ids = list(centroids_dict.keys())
                    cluster_dists = [np.linalg.norm(np.array(centroids_dict[c]) - np.array([cX, cY])) for c in cluster_ids]
                    closest_cluster_id = cluster_ids[np.argmin(cluster_dists)]

                    if min(cluster_dists) <= 20:  # Set a threshold for matching centroids
                        cluster_id = closest_cluster_id
                    else:
                        cluster_id = max(cluster_ids) + 1
                        centroids_dict[cluster_id] = (cX, cY)

                frame_positions.append((cluster_id, cX, cY))

            tracked_positions.append(frame_positions)

        return tracked_positions


    def extract_internal_porosity(contours_list, frames_df):
        porosity_list = []

        for contours, row in zip(contours_list, frames_df.itertuples()):
            frame_name = row.file_name
            frame_path = os.path.join(dataset_folder, "train", frame_name)
            frame = cv2.imread(frame_path)
            h, w, _ = frame.shape

            frame_porosities = []

            for contour in contours:
                # Convert contour to binary mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)

                # Calculate porosity based on darker pixels
                dark_pixel_count = np.sum(frame[:, :, 0] < 50)  # Assuming darker pixels have low blue channel values
                total_area = cv2.contourArea(contour)
                porosity = dark_pixel_count / total_area

                frame_porosities.append(porosity)

            porosity_list.append(frame_porosities)

        return porosity_list
    def extract_shape_similarity(contours_list, reference_shapes):
        shape_similarity_list = []

        for contours in contours_list:
            frame_shape_similarity = []

            for contour in contours:
                min_similarity = float('inf')
                closest_shape = None

                for shape_name, shape_contour in reference_shapes.items():
                    if len(contour) < 3:
                        continue

                    similarity = cv2.matchShapes(contour, shape_contour, cv2.CONTOURS_MATCH_I1, 0)
                    if similarity < min_similarity:
                        min_similarity = similarity
                        closest_shape = shape_name

                frame_shape_similarity.append(closest_shape)

            shape_similarity_list.append(frame_shape_similarity)

        return shape_similarity_list

    def save_shape_similarity_to_csv(shape_similarity_list, dataset_folder, dataset_name):
        csv_file_path = os.path.join(dataset_folder, f"{dataset_name}_shape_similarity.csv")

        with open(csv_file_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Frame"] + [f"Cluster_{i}" for i in range(len(shape_similarity_list[0]))])

            for frame_idx, shape_similarities in enumerate(shape_similarity_list):
                csv_writer.writerow([f"Frame_{frame_idx}"] + shape_similarities)

    def save_porosity_to_csv(porosity_list, dataset_folder, dataset_name):
        csv_file_path = os.path.join(dataset_folder, f"{dataset_name}_porosity.csv")

        with open(csv_file_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Frame"] + [f"Cluster_{i}" for i in range(len(porosity_list[0]))])

            for frame_idx, porosities in enumerate(porosity_list):
                csv_writer.writerow([f"Frame_{frame_idx}"] + porosities)

    def track_bright_spot(contours_list, frame):
        key_tracked_positions = []

        cluster_tracks = {}  # Dictionary to track clusters between frames

        for frame_idx, contours in enumerate(contours_list):
            frame_positions = []

            for contour in contours:
                # Convert contour to a binary mask
                mask = np.zeros_like(frame, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)

                # Extract the region of interest (ROI) from the original frame using the mask
                roi = cv2.bitwise_and(frame, mask)

                # Convert the ROI to grayscale
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Find the brightest pixel's location
                max_intensity = np.max(roi_gray)
                bright_spot_location = np.argwhere(roi_gray == max_intensity)[0]

                key_spot_x, key_spot_y = bright_spot_location[1], bright_spot_location[0]

                # Check if there is a matching cluster from the previous frame
                matching_cluster_id = None
                for cluster_id, (prev_frame_idx, prev_position) in cluster_tracks.items():
                    if frame_idx - prev_frame_idx == 1:  # Only consider clusters from the previous frame
                        prev_x, prev_y = prev_position
                        distance = np.sqrt((key_spot_x - prev_x)**2 + (key_spot_y - prev_y)**2)
                        if distance < 50:  # Set a threshold for matching clusters
                            matching_cluster_id = cluster_id
                            break

                if matching_cluster_id is None:
                    # Assign a new cluster ID
                    cluster_id = len(cluster_tracks)
                    cluster_tracks[cluster_id] = (frame_idx, (key_spot_x, key_spot_y))
                else:
                    cluster_id = matching_cluster_id

                frame_positions.append((cluster_id, key_spot_x, key_spot_y))

            key_tracked_positions.append(frame_positions)

        return key_tracked_positions






    porosity_list = extract_internal_porosity(contours_list, frames_info)
    shape_similarity_list = extract_shape_similarity(contours_list, reference_shapes)
    tracked_positions = track_clusters(new_contours_list)

    #tracked_bright_spots = track_bright_spot(contours_list, frames_info)
    # Save porosity data to a CSV file
    save_porosity_to_csv(porosity_list, dataset_folder, dataset_name)

    # Save shape similarity data to a CSV file
    save_shape_similarity_to_csv(shape_similarity_list, dataset_folder, dataset_name)

    def generate_velocity_field(tracked_positions, frame_rate):
        velocity_field = []

        for frame_idx in range(len(tracked_positions) - 1):
            frame_velocities = []

            for cluster in tracked_positions[frame_idx]:
                cluster_id, x, y = cluster

                next_frame_clusters = [c for c in tracked_positions[frame_idx + 1] if c[0] == cluster_id]
                if len(next_frame_clusters) == 0:
                    continue

                next_x, next_y = next_frame_clusters[0][1], next_frame_clusters[0][2]
                displacement = np.array([next_x - x, next_y - y])
                velocity = displacement / (1 / frame_rate)  # Velocity = displacement / time_interval

                frame_velocities.append((cluster_id, velocity[0], velocity[1]))

            velocity_field.append(frame_velocities)

        return velocity_field

    # Assuming 'tracked_positions' is the output of the 'track_clusters' function
    # and 'tracked_bright_spots' is the output of the 'track_bright_spot' function
    frame_rate = 30  # Frames per second

    # Generate velocity fields for the tracked positions and key bright spots
    position_velocity_field = generate_velocity_field(tracked_positions, frame_rate)
    #bright_spot_velocity_field = generate_velocity_field(tracked_bright_spots, frame_rate)

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Assuming you have the 'position_velocity_field' generated using the function

    def generate_average_position_velocity(velocity_field):
        position_velocities = {}

        for frame_velocities in velocity_field:
            for cluster_id, velocity_x, velocity_y in frame_velocities:
                if cluster_id not in position_velocities:
                    position_velocities[cluster_id] = []

                position_velocities[cluster_id].append((velocity_x, velocity_y))

        average_position_velocities = {}
        for cluster_id, velocities in position_velocities.items():
            average_velocity = np.mean(velocities, axis=0)
            average_position_velocities[cluster_id] = average_velocity

        return average_position_velocities

    def generate_velocity_heatmap(average_position_velocities, grid_size, frame_width, frame_height, arrow_scale=0.5):
        x_positions = []
        y_positions = []
        average_x_velocities = []
        average_y_velocities = []

        for cluster_id, (avg_velocity_x, avg_velocity_y) in average_position_velocities.items():
            x_positions.append(cluster_id % grid_size)
            y_positions.append(cluster_id // grid_size)
            average_x_velocities.append(avg_velocity_x)
            average_y_velocities.append(avg_velocity_y)

        plt.figure(figsize=(12, 8))
        plt.quiver(x_positions, y_positions, average_x_velocities, average_y_velocities,
                scale=arrow_scale, scale_units='xy', angles='xy', color='b', alpha=0.6)
        plt.xlim(0, grid_size)
        plt.ylim(0, grid_size)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Average Velocity Vectors Heatmap')
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        plt.show()


    # Define the dimensions of the frame (image) and the grid size
    frame_width = 500  # Adjust this based on the actual frame width
    frame_height = 500  # Adjust this based on the actual frame height
    grid_size = 50  # Adjust this based on the desired grid size

    # Generate average velocities for positions
    average_position_velocities = generate_average_position_velocity(position_velocity_field)

    # Adjust the arrow_scale parameter to control the length of the arrows
    arrow_scale = 50  # Change this value as needed

    # Generate and display the velocity vectors heatmap with adjusted arrow length
    generate_velocity_heatmap(average_position_velocities, grid_size, frame_width, frame_height, arrow_scale)

    cap = cv2.VideoCapture(video_source)
    for index, row in frames_info.iterrows():
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = row["file_name"]
        contours_folder_path = os.path.join("/content/drive/My Drive", "datasets", dataset_name, "contours")
        contours_frame_path = os.path.join(contours_folder_path, frame_name)

        frame_with_contours = cv2.imread(contours_frame_path)

        cv2_imshow(frame_with_contours)  # Display the frame with contours using cv2_imshow

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    from google.colab.patches import cv2_imshow
    import cv2

    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2_imshow(frame)  # Display the frame using cv2_imshow
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    contours_folder_path = os.path.join("/content/drive/My Drive", "datasets", dataset_name, "contours")

    # Choose the index of the frame you want to display
    frame_index = 1  # Change this to the desired frame index

    frame_name = frames_info.loc[frame_index, "file_name"]
    contours_frame_path = os.path.join(contours_folder_path, frame_name)

    frame_with_contours = cv2.imread(contours_frame_path)

    cv2_imshow(frame_with_contours)  # Display the frame with contours using cv2_imshow

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    import os
    import cv2
    from google.colab.patches import cv2_imshow

    contours_folder_path = os.path.join("/content/drive/My Drive", "datasets", dataset_name, "contours")

    # Choose the index of the frame you want to display
    frame_index = 1  # Change this to the desired frame index

    frame_name = frames_info.loc[frame_index, "file_name"]
    merged_contours_frame_path = os.path.join(contours_folder_path, "merged", frame_name)

    frame_with_merged_contours = cv2.imread(merged_contours_frame_path)

    cv2_imshow(frame_with_merged_contours)  # Display the frame with merged contours using cv2_imshow

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for frame_idx, frame_positions in enumerate(tracked_positions):
        print(f"Frame {frame_idx}: {frame_positions}")

    # Define the output video file
    output_video_path = os.path.join("/content/drive/My Drive", "tracked_video.avi")

    # Get the frame dimensions from the first frame
    frame_width = frames_info["frame_width"].iloc[0]
    frame_height = frames_info["frame_height"].iloc[0]

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

    # Iterate through the tracked_positions and frames_info DataFrames
    for frame_idx, (frame_positions, row) in enumerate(zip(tracked_positions, frames_info.iterrows())):
        _, frame_info = row

        frame_name = frame_info["file_name"]
        frame_path = os.path.join("/content/drive/My Drive", "datasets", dataset_name, "train", "images", frame_name)
        frame = cv2.imread(frame_path)

        for cluster_id, cX, cY in frame_positions:
            cv2.circle(frame, (cX, cY), 10, (0, 255, 0), -1)
            cv2.putText(frame, str(cluster_id), (cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out.write(frame)

    # Release the VideoWriter and close the video file
    out.release()

    print(f"Tracked video saved to: {output_video_path}")

    # Example usage:
    video_source = "/content/drive/My Drive/Data/vid.avi"
    dataset_name = "my_dataset"
    dataset_folder = os.path.join("/content/drive/My Drive", "datasets", dataset_name)
    frames_info = generate_frames(video_source, dataset_name)
    # ... other processing steps ...

    # Stitch contour images into a video
    output_video_path = os.path.join(dataset_folder, f"{dataset_name}_contours_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Choose the appropriate codec
    fps = 30  # Frames per second
    width, height = frames_info.iloc[0]["frame_width"], frames_info.iloc[0]["frame_height"]
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    contours_folder = "contours"
    contours_folder_path = os.path.join(dataset_folder, contours_folder)

    for index, row in frames_info.iterrows():
        frame_name = row["file_name"]
        frame_path = os.path.join(contours_folder_path, frame_name)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()

    import cv2
    import numpy as np

    # Video Loading
    video_source = "/content/drive/My Drive/Data/not.avi"
    output_path = "/content/drive/My Drive/Data/optical_flow_video.avi"

    # Load video
    cap = cv2.VideoCapture(video_source)

    # Get video frame properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize variables for the first frame
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0

    # Loop through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate flow magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Map angle to hue in HSV color space
        hue = (angle * 180 / np.pi) / 2  # Scale angle to 0-180 (Hue in OpenCV)
        hsv = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        hsv[..., 0] = hue
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV to BGR for visualization
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Write the frame with optical flow visualization to the output video
        out.write(bgr)

        frame_count += 1

        # Update the previous frame and gray frame for the next iteration
        prev_gray = gray

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Check if the video was saved successfully
    if os.path.exists(output_path):
        print(f"Optical flow video saved successfully to {output_path}")
    else:
        print("Error: Optical flow video was not saved")

    import cv2
    import numpy as np
    from google.colab.patches import cv2_imshow
    import matplotlib.pyplot as plt

    # Create a color wheel image
    wheel_size = 200
    color_wheel = np.zeros((wheel_size, wheel_size, 3), dtype=np.uint8)

    # Define the angles for flow directions
    angles = np.linspace(0, 2 * np.pi, wheel_size, endpoint=False)

    # Create a colormap to map angles to colors
    colormap = plt.get_cmap('hsv')

    # Map angles to colors and fill the color wheel
    for i, angle in enumerate(angles):
        color = (np.array(colormap(angle / (2 * np.pi))) * 255).astype(np.uint8)[:3]
        color = tuple(map(int, color))  # Convert color to tuple of integers
        start_angle = int(i)
        end_angle = int(i + 1)
        cv2.ellipse(color_wheel, (wheel_size // 2, wheel_size // 2), (wheel_size // 2, wheel_size // 2),
                    0, start_angle, end_angle, color, -1)

    # Show the color wheel image
    cv2_imshow(color_wheel)

    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    # Video Loading
    video_source = "/content/drive/My Drive/Data/done.avi"
    dataset_name = "my_dataset"
    dataset_folder = os.path.join("/content/drive/My Drive", "datasets", dataset_name)

    # Create the dataset folder if it doesn't exist
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Load video
    cap = cv2.VideoCapture(video_source)

    # Initialize variables for the first frame
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize variables for accumulating flow vectors
    total_flow_x = np.zeros_like(prev_gray, dtype=np.float32)
    total_flow_y = np.zeros_like(prev_gray, dtype=np.float32)

    frame_count = 0

    # Loop through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Accumulate flow vectors
        total_flow_x += flow[..., 0]
        total_flow_y += flow[..., 1]

        frame_count += 1

        # Update the previous frame and gray frame for the next iteration
        prev_gray = gray

    # Calculate average flow vectors
    average_flow_x = total_flow_x / frame_count
    average_flow_y = total_flow_y / frame_count

    # Create a figure for plotting
    plt.figure(figsize=(10, 10))

    # Scale factor for larger arrows
    scale_factor = 10

    # Draw average velocity arrows
    for y in range(0, average_flow_x.shape[0], 10):
        for x in range(0, average_flow_x.shape[1], 10):
            fx = average_flow_x[y, x]
            fy = average_flow_y[y, x]
            plt.arrow(x, y, fx * scale_factor, fy * scale_factor, head_width=5, head_length=5, color='blue')

    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()

    # Release the video capture object
    cap.release()

    import os
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    # Video Loading
    video_source = "/content/drive/My Drive/Data/done.avi"
    dataset_name = "my_dataset"
    dataset_folder = os.path.join("/content/drive/My Drive", "datasets", dataset_name)

    # Create the dataset folder if it doesn't exist
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Load video
    cap = cv2.VideoCapture(video_source)

    # Initialize variables for the first frame
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize variables for accumulating flow vectors
    total_flow_x = np.zeros_like(prev_gray, dtype=np.float32)
    total_flow_y = np.zeros_like(prev_gray, dtype=np.float32)

    frame_count = 0

    # Loop through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Accumulate flow vectors
        total_flow_x += flow[..., 0]
        total_flow_y += flow[..., 1]

        frame_count += 1

        # Update the previous frame and gray frame for the next iteration
        prev_gray = gray

    # Calculate average flow vectors
    average_flow_x = total_flow_x / frame_count
    average_flow_y = total_flow_y / frame_count

    # Define scaling factors (pixels to microns and frame rate)
    pixels_per_micron = (0.5)  # Example value, replace with your actual scaling factor
    frame_rate = 10  # Example value, replace with your actual frame rate

    # Convert average flow values to microns per second
    average_velocity_x_micron_per_sec = average_flow_x / pixels_per_micron * frame_rate
    average_position_y_micron = np.arange(average_velocity_x_micron_per_sec.shape[0]) /pixels_per_micron

    # Create a figure for plotting
    plt.figure(figsize=(10, 10))

    # Plot velocity in x-direction against position in y-direction
    plt.plot(average_velocity_x_micron_per_sec.mean(axis=1), average_position_y_micron, color='blue')

    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.xlabel('Average Velocity in X-direction (µm/s)')
    plt.ylabel('Position in Y-direction (µm)')
    plt.title('Velocity in X-direction vs. Position in Y-direction')
    plt.show()

    # Release the video capture object
    cap.release()

    import cv2
    import numpy as np
    import os
    import csv
    import pandas as pd

    # Video Loading
    video_source = "/content/drive/My Drive/Data/done startup.avi"
    dataset_name = "my_dataset"
    dataset_folder = os.path.join("/content/drive/My Drive", "datasets", dataset_name)

    # Create the dataset folder if it doesn't exist
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Load video
    cap = cv2.VideoCapture(video_source)

    # Initialize variables
    prev_frame = None
    prev_gray = None
    cluster_tracking = {}  # To store cluster history and continuity

    # Define the threshold velocity
    threshold_velocity = 5.0  # Adjust this value based on your video characteristics

    # Dictionary to store cluster information
    cluster_info = {}

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_frame = frame.copy()
            prev_gray = gray_frame.copy()
            continue

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Simulate object detection/tracking
        detected_bboxes = [(100, 50, 150, 300), (300, 200, 100, 250)]  # Example bounding boxes (x, y, width, height)

        # Update cluster history and continuity
        for bbox in detected_bboxes:
            cluster_id = f"cluster_{bbox[0]}_{bbox[1]}"  # Unique cluster identifier

            if cluster_id not in cluster_tracking:
                cluster_tracking[cluster_id] = {
                    'history': [],
                    'first_cross_center_frame': None,
                    'last_cross_center_frame': None
                }

            # Calculate object's optical flow velocity within the bbox
            x1, y1, w, h = bbox
            flow_bbox = flow[y1:y1+h, x1:x1+w, :]
            velocity_x = np.mean(flow_bbox[:, :, 0])
            velocity_y = np.mean(flow_bbox[:, :, 1])
            velocity = np.sqrt(velocity_x**2 + velocity_y**2)

            # Check if the object's edge crosses the center
            edge_crossed_center = False
            if x1 <= frame.shape[1] / 2 <= x1 + w:
                if velocity > threshold_velocity:  # Define a suitable threshold
                    edge_crossed_center = True

            # Update cluster history and continuity
            cluster_tracking[cluster_id]['history'].append(edge_crossed_center)
            collided = any(cluster_tracking[cluster_id]['history'][-5:])  # Check last 5 frames

            if collided:
                # Reset continuity count and update first_cross_center_frame
                cluster_tracking[cluster_id]['first_cross_center_frame'] = None
                cluster_tracking[cluster_id]['last_cross_center_frame'] = None
            else:
                if edge_crossed_center:
                    if cluster_tracking[cluster_id]['first_cross_center_frame'] is None:
                        cluster_tracking[cluster_id]['first_cross_center_frame'] = frame_index
                    cluster_tracking[cluster_id]['last_cross_center_frame'] = frame_index

        # Update previous frame and gray frame
        prev_frame = frame.copy()
        prev_gray = gray_frame.copy()

        frame_index += 1

    # Save cluster information to an Excel file
    excel_file_path = os.path.join(dataset_folder, "cluster_info.xlsx")

    # Create a DataFrame from the cluster_info dictionary
    cluster_info_df = pd.DataFrame.from_dict(cluster_info, orient='index', columns=['length', 'continuity'])

    # Save the DataFrame to an Excel file
    cluster_info_df.to_excel(excel_file_path, index_label='cluster_id')

    # Release the video capture
    cap.release()