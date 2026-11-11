# vision/VectorizedDeepSort.py
"""
High-Precision, Stabilized Vectorized DeepSORT Multi-Object Tracker.
Key Accuracy & Stability Enhancements:
1. Priority Matching Cascade (Active tracks match before occluded tracks) with Hard Spatial Radius
   and Scale Consistency Gating to eliminate cross-frame ID jumps.
2. Velocity-Clamped & Damped 8D Kalman Filter with Frozen Occlusion Covariance:
   - Prevents covariance explosion on occluded tracks.
   - Zeroes aspect-ratio velocity (va) and damps velocity on missed frames so boxes never drift
     left/right/up/down when a person stops or exits the frame.
3. Adaptive EMA Bounding-Box Smoothing (smooth_ltrb) for jitter-free bounding boxes.
4. Appearance-Vetoed Stage B1/B2 IoU Association to prevent ID swaps when people cross paths in crowds.
5. Track-Level Duplicate Suppression (NMS across active tracks) + High-Confidence Track Initiation
   so multiple bounding boxes never pile up on the same person.
"""
from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from core.ports import IMultiObjectTracker


class VectorizedTrack:
    """Represents a single tracked target with 8D Kalman state [cx, cy, a, h, vx, vy, va, vh]."""

    __slots__ = (
        "track_id",
        "mean",
        "covariance",
        "hits",
        "age",
        "time_since_update",
        "state",
        "ema_feat",
        "gallery",
        "det_conf",
        "nn_budget",
        "smooth_ltrb",
    )

    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3

    def __init__(
        self,
        track_id: int,
        mean: np.ndarray,
        covariance: np.ndarray,
        feat: np.ndarray,
        conf: float,
        nn_budget: int = 30,
    ):
        self.track_id = track_id
        self.mean = mean
        self.covariance = covariance
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = self.TENTATIVE

        norm = float(np.linalg.norm(feat))
        unit_feat = (feat / norm).astype(np.float32) if norm > 1e-6 else feat.astype(np.float32)
        self.ema_feat = unit_feat.copy()
        self.gallery = unit_feat.reshape(1, -1).copy()
        self.det_conf = conf
        self.nn_budget = nn_budget
        self.smooth_ltrb = self.raw_kalman_ltrb()

    def is_tentative(self) -> bool:
        return self.state == self.TENTATIVE

    def is_confirmed(self) -> bool:
        return self.state == self.CONFIRMED

    def is_deleted(self) -> bool:
        return self.state == self.DELETED

    def raw_kalman_ltrb(self) -> list[float]:
        cx, cy, a, h = self.mean[:4]
        h = max(16.0, float(h))
        a = max(0.12, min(3.0, float(a)))
        w = a * h
        half_w = w * 0.5
        half_h = h * 0.5
        return [float(cx - half_w), float(cy - half_h), float(cx + half_w), float(cy + half_h)]

    def to_ltrb(self) -> list[float]:
        return list(self.smooth_ltrb)

    def to_ltwh(self) -> list[float]:
        x1, y1, x2, y2 = self.smooth_ltrb
        return [float(x1), float(y1), float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1))]

    def sync_smooth_box(self, det_ltrb: Optional[np.ndarray] = None, alpha: float = 0.65) -> None:
        """
        Update stabilized bounding box:
        - On matched frames: blends detector measurement with Kalman filter state and applies
          motion-adaptive EMA smoothing to eliminate jitter without lagging fast walkers.
        - On coasting frames: applies heavily damped velocity shift so boxes never fly off.
        """
        k_box = np.asarray(self.raw_kalman_ltrb(), dtype=np.float64)
        prev_box = np.asarray(self.smooth_ltrb, dtype=np.float64)

        if det_ltrb is not None:
            d_box = np.asarray(det_ltrb, dtype=np.float64)
            target = 0.72 * d_box + 0.28 * k_box
            # Motion-adaptive alpha: if person moved significantly, follow target more closely
            h = max(16.0, float(target[3] - target[1]))
            center_shift = np.hypot(
                0.5 * (target[0] + target[2] - prev_box[0] - prev_box[2]),
                0.5 * (target[1] + target[3] - prev_box[1] - prev_box[3]),
            )
            eff_alpha = min(0.92, alpha + 0.35 * min(1.0, center_shift / (0.12 * h)))
            smoothed = eff_alpha * target + (1.0 - eff_alpha) * prev_box
            self.smooth_ltrb = [float(x) for x in smoothed]
        elif self.time_since_update == 1:
            vx = float(self.mean[4]) * 0.25
            vy = float(self.mean[5]) * 0.25
            self.smooth_ltrb = [
                float(prev_box[0] + vx),
                float(prev_box[1] + vy),
                float(prev_box[2] + vx),
                float(prev_box[3] + vy),
            ]

    def get_feature(self) -> np.ndarray:
        return self.ema_feat

    def update_appearance(self, feat: np.ndarray, is_clean: bool = True) -> None:
        if not is_clean or feat is None:
            return
        norm = float(np.linalg.norm(feat))
        if norm <= 1e-6:
            return
        unit_feat = (feat / norm).astype(np.float32)

        # Reject sudden appearance outliers (e.g. partial occlusion by another person)
        sim_to_ema = float(np.dot(self.ema_feat, unit_feat))
        if self.hits >= 3 and sim_to_ema < 0.55:
            return

        ema = 0.85 * self.ema_feat + 0.15 * unit_feat
        ema_norm = float(np.linalg.norm(ema))
        if ema_norm > 1e-6:
            self.ema_feat = (ema / ema_norm).astype(np.float32)

        if len(self.gallery) >= self.nn_budget:
            self.gallery = np.roll(self.gallery, -1, axis=0)
            self.gallery[-1] = unit_feat
        else:
            self.gallery = np.vstack((self.gallery, unit_feat.reshape(1, -1)))


class VectorizedDeepSort(IMultiObjectTracker):
    """
    Vectorized DeepSORT implementation with Priority Matching Cascade, Spatial/Scale Gating,
    Velocity-Clamped Kalman Filtering, and Duplicate Track Suppression.
    """

    CHI2_INV_95_4D = 9.4877
    INFTY_COST = 1e5

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.55,
        max_cosine_distance: float = 0.26,
        nn_budget: int = 30,
        conf_high: float = 0.45,
        conf_low: float = 0.28,
        init_conf: float = 0.52,
        box_smooth_alpha: float = 0.65,
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.init_conf = max(conf_high, init_conf)
        self.box_smooth_alpha = box_smooth_alpha

        self.tracks: list[VectorizedTrack] = []
        self._next_id = 1
        self._std_pos = 1.0 / 20.0
        self._std_vel = 1.0 / 160.0

    @staticmethod
    def _boxes_to_xyah(boxes_xyxy: np.ndarray) -> np.ndarray:
        if len(boxes_xyxy) == 0:
            return np.empty((0, 4), dtype=np.float64)
        w = np.maximum(1.0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
        h = np.maximum(1.0, boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        cx = boxes_xyxy[:, 0] + w * 0.5
        cy = boxes_xyxy[:, 1] + h * 0.5
        return np.column_stack((cx, cy, w / h, h)).astype(np.float64)

    @staticmethod
    def _compute_iou_and_containment(
        boxes_a: np.ndarray, boxes_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (iou_matrix, io_min_matrix) between (K, 4) and (M, 4) boxes in [x1, y1, x2, y2]."""
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            empty = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
            return empty, empty
        tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
        br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
        wh = np.maximum(0.0, br - tl)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area_a = np.maximum(1.0, (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))[:, None]
        area_b = np.maximum(1.0, (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))[None, :]
        iou = inter / np.maximum(1e-6, area_a + area_b - inter)
        io_min = inter / np.maximum(1e-6, np.minimum(area_a, area_b))
        return iou, io_min

    @classmethod
    def _compute_iou_matrix(cls, boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Vectorized pairwise IoU between (K, 4) and (M, 4) boxes in [x1, y1, x2, y2] format."""
        iou, _ = cls._compute_iou_and_containment(boxes_a, boxes_b)
        return iou

    def _predict_all(self) -> None:
        if not self.tracks:
            return
        means = np.array([t.mean for t in self.tracks], dtype=np.float64)
        covs = np.array([t.covariance for t in self.tracks], dtype=np.float64)

        # Always force aspect-ratio velocity (va) to zero so boxes never stretch on their own
        means[:, 6] = 0.0

        for i, t in enumerate(self.tracks):
            if t.time_since_update == 1:
                # Strongly damp velocity on first missed frame
                means[i, 4:6] *= 0.25
                means[i, 6:8] = 0.0
            elif t.time_since_update >= 2:
                # Freeze velocity completely once occluded so track never drifts across the frame
                means[i, 4:8] = 0.0
            t.age += 1
            t.time_since_update += 1

        # Update positions using damped velocities
        means[:, :4] += means[:, 4:]
        means[:, 2] = np.clip(means[:, 2], 0.15, 2.80)
        means[:, 3] = np.maximum(16.0, means[:, 3])

        # Only grow covariance for actively visible / just-missed tracks (prevents covariance explosion)
        active_cov_mask = np.array([t.time_since_update <= 2 for t in self.tracks], dtype=bool)
        if np.any(active_cov_mask):
            h = means[active_cov_mask, 3]
            p = (self._std_pos * h) ** 2
            v = (self._std_vel * h) ** 2
            c_sub = covs[active_cov_mask]
            c44 = c_sub[:, 4:, 4:]
            c_sub[:, :4, :4] += c_sub[:, :4, 4:] + c_sub[:, 4:, :4] + c44
            c_sub[:, :4, 4:] += c44
            c_sub[:, 4:, :4] += c44
            c_sub[:, 0, 0] += p
            c_sub[:, 1, 1] += p
            c_sub[:, 2, 2] += 1e-4
            c_sub[:, 3, 3] += p
            c_sub[:, 4, 4] += v
            c_sub[:, 5, 5] += v
            c_sub[:, 6, 6] += 1e-10
            c_sub[:, 7, 7] += v
            covs[active_cov_mask] = c_sub

        for i, t in enumerate(self.tracks):
            t.mean = means[i]
            t.covariance = covs[i]
            if t.time_since_update == 1:
                t.sync_smooth_box(det_ltrb=None, alpha=self.box_smooth_alpha)

    def _kalman_update_batch(
        self,
        matches: list[tuple[int, int]],
        det_boxes_xyxy: np.ndarray,
        det_xyah: np.ndarray,
        confs: np.ndarray,
        embeds: Optional[np.ndarray],
        clean_mask: Optional[np.ndarray],
        is_low_conf: bool = False,
    ) -> None:
        if not matches:
            return
        m_tidx = [m[0] for m in matches]
        m_didx = [m[1] for m in matches]

        m_means = np.array([self.tracks[i].mean for i in m_tidx], dtype=np.float64)
        m_covs = np.array([self.tracks[i].covariance for i in m_tidx], dtype=np.float64)
        p = (self._std_pos * m_means[:, 3]) ** 2
        proj_covs = m_covs[:, :4, :4].copy()
        proj_covs[:, 0, 0] += p
        proj_covs[:, 1, 1] += p
        proj_covs[:, 2, 2] += 1e-2
        proj_covs[:, 3, 3] += p

        kg = np.linalg.solve(proj_covs, m_covs[:, :, :4].transpose(0, 2, 1)).transpose(0, 2, 1)
        innov = det_xyah[m_didx] - m_means[:, :4]
        new_means = m_means + np.einsum("kij,kj->ki", kg, innov)
        new_covs = m_covs - np.einsum("kij,kjl->kil", kg, m_covs[:, :4, :])

        for idx, (tidx, didx) in enumerate(matches):
            tr = self.tracks[tidx]
            prev_tsu = tr.time_since_update
            nm = new_means[idx]

            # Always zero aspect-ratio velocity and clamp linear velocity to realistic human bounds
            nm[2] = float(np.clip(nm[2], 0.15, 2.80))
            nm[3] = max(16.0, float(nm[3]))
            nm[6] = 0.0
            nm[7] = float(np.clip(nm[7] * 0.25, -0.02 * nm[3], 0.02 * nm[3]))

            if is_low_conf or prev_tsu > 2:
                # Do not build high velocity from a low-conf recovery or post-occlusion re-association
                nm[4] = float(m_means[idx, 4] * 0.25)
                nm[5] = float(m_means[idx, 5] * 0.25)
            else:
                max_v = (0.045 if tr.is_confirmed() else 0.025) * nm[3]
                speed = float(np.hypot(nm[4], nm[5]))
                if speed > max_v and speed > 1e-6:
                    scale = max_v / speed
                    nm[4] *= scale
                    nm[5] *= scale

            tr.mean = nm
            tr.covariance = new_covs[idx]
            tr.time_since_update = 0
            tr.hits += 1
            c_val = float(confs[didx])
            tr.det_conf = c_val
            tr.sync_smooth_box(det_boxes_xyxy[didx], alpha=self.box_smooth_alpha)

            if tr.state == VectorizedTrack.TENTATIVE and tr.hits >= self.n_init:
                tr.state = VectorizedTrack.CONFIRMED
            if not is_low_conf and embeds is not None and len(embeds) > didx:
                is_clean = (c_val >= self.conf_high) and (clean_mask[didx] if clean_mask is not None else True)
                tr.update_appearance(embeds[didx], is_clean=is_clean)

    def _match_stage_a_subset(
        self,
        track_indices: list[int],
        det_indices: list[int],
        high_boxes: np.ndarray,
        high_xyah: np.ndarray,
        high_embeds: np.ndarray,
        strict_occlusion: bool = False,
    ) -> tuple[list[tuple[int, int]], list[int]]:
        if not track_indices or not det_indices:
            return [], det_indices

        K = len(track_indices)
        M = len(det_indices)
        sub_xyah = high_xyah[det_indices]
        sub_boxes = high_boxes[det_indices].astype(np.float64)
        sub_embeds = high_embeds[det_indices]

        t_means = np.array([self.tracks[i].mean[:4] for i in track_indices], dtype=np.float64)
        t_covs = np.array([self.tracks[i].covariance[:4, :4] for i in track_indices], dtype=np.float64)
        p = (self._std_pos * t_means[:, 3]) ** 2
        t_covs[:, 0, 0] += p
        t_covs[:, 1, 1] += p
        t_covs[:, 2, 2] += 1e-2
        t_covs[:, 3, 3] += p

        diffs = (sub_xyah[None, :, :] - t_means[:, None, :]).transpose(0, 2, 1)
        sol = np.linalg.solve(t_covs, diffs)
        maha = np.einsum("kij,kij->kj", diffs, sol)

        t_boxes_all = np.array([self.tracks[i].to_ltrb() for i in track_indices], dtype=np.float64)
        iou_mat = self._compute_iou_matrix(t_boxes_all, sub_boxes)

        # Hard spatial distance & scale consistency gates
        dx = sub_xyah[None, :, 0] - t_means[:, None, 0]
        dy = sub_xyah[None, :, 1] - t_means[:, None, 1]
        center_dist = np.hypot(dx, dy)

        t_h = np.maximum(16.0, t_means[:, 3:4])
        d_h = np.maximum(16.0, sub_xyah[None, :, 3])
        scale_ratio = np.minimum(t_h, d_h) / np.maximum(t_h, d_h)

        thresh = self.max_cosine_distance * (0.85 if strict_occlusion else 1.0)
        cost = np.empty((K, M), dtype=np.float32)

        for r, tidx in enumerate(track_indices):
            tr = self.tracks[tidx]
            min_d = 1.0 - (tr.gallery @ sub_embeds.T).max(axis=0)
            ema_d = 1.0 - (tr.ema_feat @ sub_embeds.T)
            app_cost = np.maximum(0.0, 0.60 * min_d + 0.40 * ema_d)
            age_penalty = min(0.03, (tr.time_since_update - 1) * 0.005)
            cost[r] = np.maximum(0.0, app_cost - 0.04 * iou_mat[r]) + age_penalty

            # Maximum physical displacement allowed in local tracker
            max_radius = float(t_h[r, 0]) * (0.55 + 0.12 * min(tr.time_since_update, 4))
            invalid = (
                (maha[r] > self.CHI2_INV_95_4D)
                | (scale_ratio[r] < 0.58)
                | (center_dist[r] > max_radius)
            )
            cost[r, invalid] = self.INFTY_COST

        cost_gated = np.where(cost > thresh, thresh + 1e-5, cost)
        row_ind, col_ind = linear_sum_assignment(cost_gated)

        matches: list[tuple[int, int]] = []
        rem_dets = set(range(M))
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= thresh:
                matches.append((track_indices[r], det_indices[c]))
                rem_dets.discard(c)

        unmatched_dets = [det_indices[c] for c in sorted(rem_dets)]
        return matches, unmatched_dets

    def _suppress_duplicate_tracks(self) -> None:
        """
        Track-level NMS: if two active tracks overlap heavily on the same person,
        retain the more established track and mark the duplicate as DELETED.
        """
        alive_idx = [i for i, t in enumerate(self.tracks) if t.state != VectorizedTrack.DELETED]
        if len(alive_idx) <= 1:
            return

        # Sort by priority: CONFIRMED before TENTATIVE, lower time_since_update, higher hits, lower track_id
        alive_idx.sort(
            key=lambda i: (
                0 if self.tracks[i].is_confirmed() else 1,
                self.tracks[i].time_since_update,
                -self.tracks[i].hits,
                self.tracks[i].track_id,
            )
        )

        boxes = np.array([self.tracks[i].to_ltrb() for i in alive_idx], dtype=np.float64)
        hs = np.maximum(16.0, boxes[:, 3] - boxes[:, 1])
        cxs = 0.5 * (boxes[:, 0] + boxes[:, 2])
        ws = np.maximum(16.0, boxes[:, 2] - boxes[:, 0])
        iou_mat, io_min_mat = self._compute_iou_and_containment(boxes, boxes)

        suppressed: set[int] = set()
        for a_pos in range(len(alive_idx)):
            if a_pos in suppressed:
                continue
            for b_pos in range(a_pos + 1, len(alive_idx)):
                if b_pos in suppressed:
                    continue
                iou_val = iou_mat[a_pos, b_pos]
                io_min_val = io_min_mat[a_pos, b_pos]
                scale_sim = min(hs[a_pos], hs[b_pos]) / max(hs[a_pos], hs[b_pos])
                dx_rel = abs(cxs[a_pos] - cxs[b_pos]) / max(ws[a_pos], ws[b_pos])

                if iou_val > 0.46 or (io_min_val > 0.68 and scale_sim > 0.52 and dx_rel < 0.30):
                    suppressed.add(b_pos)
                    self.tracks[alive_idx[b_pos]].state = VectorizedTrack.DELETED

    def update(self, boxes_xyxy: np.ndarray, confs: np.ndarray, embeds: np.ndarray) -> list[VectorizedTrack]:
        """
        Execute one full stabilized multi-target tracking step.
        """
        self._predict_all()

        if len(boxes_xyxy) == 0:
            for tr in self.tracks:
                if tr.state == VectorizedTrack.TENTATIVE or tr.time_since_update > self.max_age:
                    tr.state = VectorizedTrack.DELETED
            self.tracks = [t for t in self.tracks if t.state != VectorizedTrack.DELETED]
            return self.tracks

        high_mask = confs >= self.conf_high
        low_mask = (confs >= self.conf_low) & (~high_mask)

        high_boxes = boxes_xyxy[high_mask]
        high_confs = confs[high_mask]
        high_embeds = embeds[high_mask]
        high_xyah = self._boxes_to_xyah(high_boxes)

        low_boxes = boxes_xyxy[low_mask]
        low_confs = confs[low_mask]
        low_embeds = embeds[low_mask]
        low_xyah = self._boxes_to_xyah(low_boxes)

        # Identify isolated detections (no overlap with other detections) for clean gallery updates
        if len(high_boxes) > 1:
            det_iou, det_iomin = self._compute_iou_and_containment(high_boxes, high_boxes)
            np.fill_diagonal(det_iou, 0.0)
            np.fill_diagonal(det_iomin, 0.0)
            high_clean = (det_iou.max(axis=1) < 0.25) & (det_iomin.max(axis=1) < 0.45)
        else:
            high_clean = np.ones(len(high_boxes), dtype=bool)

        active_confirmed_idx = [
            i for i, t in enumerate(self.tracks)
            if t.state == VectorizedTrack.CONFIRMED and t.time_since_update <= 2
        ]
        occluded_confirmed_idx = [
            i for i, t in enumerate(self.tracks)
            if t.state == VectorizedTrack.CONFIRMED and t.time_since_update > 2
        ]
        unconfirmed_idx = [i for i, t in enumerate(self.tracks) if t.state == VectorizedTrack.TENTATIVE]

        matches_high: list[tuple[int, int]] = []
        unmatched_high_dets = list(range(len(high_boxes)))
        matched_tracks_set: set[int] = set()

        # === STAGE A1: Priority Matching for Currently Active Confirmed Tracks (tsu <= 2) ===
        if active_confirmed_idx and unmatched_high_dets:
            m_a1, unmatched_high_dets = self._match_stage_a_subset(
                active_confirmed_idx,
                unmatched_high_dets,
                high_boxes,
                high_xyah,
                high_embeds,
                strict_occlusion=False,
            )
            for tidx, didx in m_a1:
                matches_high.append((tidx, didx))
                matched_tracks_set.add(tidx)

        # === STAGE B1: IoU + Appearance Gated Matching for Just-Missed Confirmed & Tentative Tracks ===
        iou_candidates = [
            i for i in active_confirmed_idx
            if i not in matched_tracks_set and self.tracks[i].time_since_update == 1
        ] + [i for i in unconfirmed_idx if self.tracks[i].time_since_update == 1]

        if iou_candidates and unmatched_high_dets:
            t_boxes = np.array([self.tracks[i].to_ltrb() for i in iou_candidates], dtype=np.float64)
            d_boxes = high_boxes[unmatched_high_dets].astype(np.float64)
            d_embs = high_embeds[unmatched_high_dets]

            iou_mat = self._compute_iou_matrix(t_boxes, d_boxes)
            cost_iou = 1.0 - iou_mat

            t_h = np.maximum(16.0, t_boxes[:, 3:4] - t_boxes[:, 1:2])
            d_h = np.maximum(16.0, (d_boxes[:, 3] - d_boxes[:, 1])[None, :])
            scale_ratio = np.minimum(t_h, d_h) / np.maximum(t_h, d_h)

            app_veto_thresh = self.max_cosine_distance * 1.15
            for r, tidx in enumerate(iou_candidates):
                tr = self.tracks[tidx]
                app_d = 1.0 - (tr.ema_feat @ d_embs.T)
                invalid = (app_d > app_veto_thresh) | (scale_ratio[r] < 0.68)
                cost_iou[r, invalid] = self.INFTY_COST

            cost_gated = np.where(cost_iou > self.max_iou_distance, self.max_iou_distance + 1e-5, cost_iou)
            row_ind, col_ind = linear_sum_assignment(cost_gated)
            rem_dets = set(range(len(unmatched_high_dets)))
            for r, c in zip(row_ind, col_ind):
                if cost_iou[r, c] <= self.max_iou_distance:
                    tidx = iou_candidates[r]
                    didx = unmatched_high_dets[c]
                    matches_high.append((tidx, didx))
                    matched_tracks_set.add(tidx)
                    rem_dets.discard(c)
            unmatched_high_dets = [unmatched_high_dets[c] for c in sorted(rem_dets)]

        # === STAGE A2: Secondary Recovery for Occluded Confirmed Tracks (tsu > 2) ===
        if occluded_confirmed_idx and unmatched_high_dets:
            m_a2, unmatched_high_dets = self._match_stage_a_subset(
                occluded_confirmed_idx,
                unmatched_high_dets,
                high_boxes,
                high_xyah,
                high_embeds,
                strict_occlusion=True,
            )
            for tidx, didx in m_a2:
                matches_high.append((tidx, didx))
                matched_tracks_set.add(tidx)

        # Update Kalman states for all High-Confidence matches
        self._kalman_update_batch(
            matches_high, high_boxes, high_xyah, high_confs, high_embeds, high_clean, is_low_conf=False
        )

        # === STAGE B2: ByteTrack-Style Low-Confidence IoU + Appearance Recovery (tsu == 1 only) ===
        remaining_confirmed = [
            i for i in active_confirmed_idx
            if i not in matched_tracks_set and self.tracks[i].time_since_update == 1
        ]
        matches_low: list[tuple[int, int]] = []
        if remaining_confirmed and len(low_boxes) > 0:
            t_boxes = np.array([self.tracks[i].to_ltrb() for i in remaining_confirmed], dtype=np.float64)
            l_boxes = low_boxes.astype(np.float64)
            cost_iou = 1.0 - self._compute_iou_matrix(t_boxes, l_boxes)
            low_iou_thresh = 0.45  # Requires IoU >= 0.55 for low-confidence recovery

            t_h = np.maximum(16.0, t_boxes[:, 3:4] - t_boxes[:, 1:2])
            l_h = np.maximum(16.0, (l_boxes[:, 3] - l_boxes[:, 1])[None, :])
            scale_ratio = np.minimum(t_h, l_h) / np.maximum(t_h, l_h)

            for r, tidx in enumerate(remaining_confirmed):
                tr = self.tracks[tidx]
                app_d = 1.0 - (tr.ema_feat @ low_embeds.T)
                invalid = (app_d > self.max_cosine_distance * 1.10) | (scale_ratio[r] < 0.75)
                cost_iou[r, invalid] = self.INFTY_COST

            cost_gated = np.where(cost_iou > low_iou_thresh, low_iou_thresh + 1e-5, cost_iou)
            row_ind, col_ind = linear_sum_assignment(cost_gated)
            for r, c in zip(row_ind, col_ind):
                if cost_iou[r, c] <= low_iou_thresh:
                    tidx = remaining_confirmed[r]
                    matches_low.append((tidx, c))
                    matched_tracks_set.add(tidx)

        if matches_low:
            self._kalman_update_batch(
                matches_low, low_boxes, low_xyah, low_confs, embeds=None, clean_mask=None, is_low_conf=True
            )

        # === Mark Missed Tracks ===
        for idx, tr in enumerate(self.tracks):
            if idx not in matched_tracks_set:
                if tr.state == VectorizedTrack.TENTATIVE or tr.time_since_update > self.max_age:
                    tr.state = VectorizedTrack.DELETED

        # === Initiate New Tracks from Unmatched High-Confidence Detections ===
        for didx in unmatched_high_dets:
            if float(high_confs[didx]) < self.init_conf:
                continue

            active_boxes = (
                np.array([t.to_ltrb() for t in self.tracks if t.state != VectorizedTrack.DELETED], dtype=np.float64)
                if self.tracks
                else np.empty((0, 4), dtype=np.float64)
            )
            if len(active_boxes) > 0:
                cand_box = high_boxes[didx : didx + 1].astype(np.float64)
                iou_ov, iomin_ov = self._compute_iou_and_containment(active_boxes, cand_box)
                if iou_ov.max() > 0.38 or iomin_ov.max() > 0.55:
                    continue

            meas = high_xyah[didx]
            mean = np.zeros(8, dtype=np.float64)
            mean[:4] = meas
            h = meas[3]
            p = (2.0 * self._std_pos * h) ** 2
            v = (2.5 * self._std_vel * h) ** 2
            cov = np.diag([p, p, 1e-4, p, v, v, 1e-10, v])
            new_track = VectorizedTrack(
                track_id=self._next_id,
                mean=mean,
                covariance=cov,
                feat=high_embeds[didx],
                conf=float(high_confs[didx]),
                nn_budget=self.nn_budget,
            )
            self.tracks.append(new_track)
            self._next_id += 1

        # Suppress any overlapping duplicate tracks before returning
        self._suppress_duplicate_tracks()
        self.tracks = [t for t in self.tracks if t.state != VectorizedTrack.DELETED]
        return self.tracks
