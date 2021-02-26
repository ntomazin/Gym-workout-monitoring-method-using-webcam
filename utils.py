
def is_in_ROI(key_points:dict, ROI_coordinates:tuple, num_of_values:list):
    #method that checks if the user is in the ROI so it can start with the measurements
    # ROI ((200, 50), (500, 450)) ((x1,y1),(x2,y2))
    ROI_x = (ROI_coordinates[0][0], ROI_coordinates[1][0])
    ROI_y = (ROI_coordinates[0][1], ROI_coordinates[1][1])
    values = [key_points[x] for x in key_points.keys() if x in num_of_values]
    for point in values:
        if point[0] < ROI_x[0] or point[0] > ROI_x[1]:
            return False
        if point[1] < ROI_y[0] or point[1] > ROI_y[1]:
            return False
    return True
