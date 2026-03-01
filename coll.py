import math
from utils.config import ROBOT_RADIUS

class ObstacleDetector:
    def compute_stop_distance(self, velocity, deceleration, time_reaction,SAFETY_MARGIN):
        """
        Compute dynamic stopping distance:
        d_stop = v^2 / 2a + reaction_time * v + margin + robot_radius
        """
        d_stop = (velocity ** 2) / (2 * deceleration)
        return d_stop + time_reaction * velocity + SAFETY_MARGIN + ROBOT_RADIUS

    def detect_obstacles(self, lidar_points, velocity, min_points=3):
        """
        lidar_points: list of (x, y) in robot frame
        Returns True if obstacle detected
        """
        stop_distance = self.compute_stop_distance(velocity)
        obstacles = []

        for x, y in lidar_points:
            if x <= 0:
                continue
            if abs(y) <= ROBOT_RADIUS and x <= stop_distance:
                obstacles.append((x, y))

        if len(obstacles) >= min_points:
            return True
        return False
