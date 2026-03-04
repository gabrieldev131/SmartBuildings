# model/tracker.py
import cv2
import numpy as np
import time
from collections import deque
from model.DataStructures.BoundingBox import BoundingBox
from model.DataStructures.StatefulTimer import StatefulTimer
import config


class StatefulTracker:
    _next_id = 0

    def __init__(self, initial_frame_gray, initial_box_tuple):
        self.id = StatefulTracker._next_id
        StatefulTracker._next_id += 1

        self._feature_params = dict(
            maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self._lk_params = dict(
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.box = BoundingBox(*initial_box_tuple)
        self.history = deque(maxlen=100)
        self.last_gray_frame = initial_frame_gray
        self.points_to_track = self._find_initial_points(initial_frame_gray, self.box)

        self.is_stopped = False
        self.stopped_timer = StatefulTimer()

    def _find_initial_points(self, frame_gray, box):
        x, y, w, h = box.to_tuple()
        roi_gray = frame_gray[y:y + h, x:x + w]
        if roi_gray.size == 0:
            return None
        points = cv2.goodFeaturesToTrack(roi_gray, mask=None, **self._feature_params)
        if points is not None:
            points[:, 0, 0] += x
            points[:, 0, 1] += y
        return points

    def get_centroid(self):
        return (self.box.x + self.box.width / 2, self.box.y + self.box.height / 2)

    def update(self, new_frame_gray, stop_thresh_px, breakout_thresh_px,
               stop_thresh_sec, smoothing_factor):
        if self.points_to_track is None or len(self.points_to_track) == 0:
            self.last_gray_frame = new_frame_gray
            return (True, self.box.to_tuple(), self.is_stopped)

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.last_gray_frame, new_frame_gray,
            self.points_to_track, None, **self._lk_params)

        if new_points is None:
            self.last_gray_frame = new_frame_gray
            return (True, self.box.to_tuple(), self.is_stopped)

        good_new = new_points[status == 1]
        good_old = self.points_to_track[status == 1]

        if len(good_new) > 1:
            dx = float(np.median([p[0] - q[0] for p, q in zip(good_new, good_old)]))
            dy = float(np.median([p[1] - q[1] for p, q in zip(good_new, good_old)]))
            calculated_box = self.box.shift(dx, dy)
            new_x = (self.box.x * smoothing_factor) + (calculated_box.x * (1 - smoothing_factor))
            new_y = (self.box.y * smoothing_factor) + (calculated_box.y * (1 - smoothing_factor))
            self.box = BoundingBox(new_x, new_y, self.box.width, self.box.height)

        self.points_to_track = good_new.reshape(-1, 1, 2)
        self.history.append((time.time(), self.get_centroid()))
        self.last_gray_frame = new_frame_gray
        self._check_if_stopped(stop_thresh_px, breakout_thresh_px, stop_thresh_sec)
        return (True, self.box.to_tuple(), self.is_stopped)

    def _check_if_stopped(self, stop_threshold, breakout_threshold, seconds_threshold):
        distance = self._calculate_movement_distance()
        if distance is None:
            return
        if self.is_stopped:
            self._process_stopped_state(distance, breakout_threshold)
        else:
            self._process_moving_state(distance, stop_threshold, seconds_threshold)

    def _calculate_movement_distance(self):
        current_time = time.time()
        recent = [item for item in self.history if current_time - item[0] <= 2.0]
        if len(recent) < 2:
            return None
        s, e = recent[0][1], recent[-1][1]
        return float(np.sqrt((s[0]-e[0])**2 + (s[1]-e[1])**2))

    def _process_stopped_state(self, distance, breakout_threshold):
        if distance > breakout_threshold:
            self._transition_to_moving()

    def _process_moving_state(self, distance, stop_threshold, seconds_threshold):
        if distance >= stop_threshold:
            self.stopped_timer.reset()
            return
        self.stopped_timer.start_if_needed()
        if self.stopped_timer.has_exceeded(seconds_threshold):
            self._transition_to_stopped()

    def _transition_to_moving(self):
        self.is_stopped = False
        self.stopped_timer.reset()

    def _transition_to_stopped(self):
        self.is_stopped = True
        # CORRECAO: config.TEST = False removido -- causava AttributeError

    def get_time(self):
        return self.stopped_timer

    def get_state(self):
        return self.is_stopped

    def reload_time(self, time_elapsed):
        self.stopped_timer = time_elapsed

    def reload_state(self, state):
        self.is_stopped = state