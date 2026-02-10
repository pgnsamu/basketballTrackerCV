import cv2
import numpy as np
from .drawPoint import PointDrawer
from utils import Player, Ball, bgr_to_hex


class DrawWindow:
    """
    Method: Majority Vote + Lock (stable team colors)
    - For each player (track_id), we collect jersey color samples.
    - During warmup (first N frames) we only collect samples and build a stable separation.
    - After warmup, we lock each track_id to a team color to avoid drifting.
    - New tracks appearing later: collect a few samples then lock quickly.

    Notes:
    - Preferred: KMeans clustering in LAB space (more stable).
    - Fallback: simple threshold on (R - B).    
    """

    def __init__(self, window_name: str):
        self.window_name = window_name
        self.point_drawer = PointDrawer(point_color="#FF0000", point_radius=7)

        self.picture_in_picture_section = None
        self.other_point = None

        # minimap overlay ölçüleri için scale değerleri
        self.scaleBig = 0.25
        self.scaleSmall = 1.0

       # --- TEAM COLORS (BGR) ---
        # Team 1 -> blue, Team 2 -> white
        self.team_colors = {
            1: (255, 0, 0),
            2: (255, 255, 255),
        }

        # track_id -> [bgr samples]
        self.jersey_samples_by_track: dict[int, list[tuple[int, int, int]]] = {}

        # track_id -> median bgr
        self.jersey_median_by_track: dict[int, tuple[int, int, int]] = {}

        # track_id -> votes
        self.team_votes_by_track: dict[int, dict[int, int]] = {}

        # track_id -> locked team_id
        self.team_id_by_track: dict[int, int] = {}

        # fallback için global threshold (R - B)
        self.score_threshold: float | None = None

        # KMeans ile öğrenilen LAB takım merkezleri
        self.team_centers_lab: np.ndarray | None = None  # (2,3)
        self.team_cluster_to_id: dict[int, int] = {}     # cluster_idx -> team_id

        
        self.warmup_frames = 60            # başta renk toplama
        self.samples_per_track = 25        # her oyuncu için max sample
        self.lock_after_votes = 40         # vote ile kilitlemede min oy sayısı
        self.lock_margin = 8               # kilitleme için oy farkı
        self.new_track_lock_samples = 6    # warmup sonrası yeni track için hızlı kilitleme

        self._warmup_done = False

    # ----------------------------
    # Jersey sampling
    # ----------------------------
    def _get_jersey_sample(self, player: Player, frame: np.ndarray) -> tuple[int, int, int]:
        """
        Eğer Player içinde ROI tabanlı renk alma fonksiyonu varsa onu dene.
        Yoksa klasik dominant renk fonksiyonuna düş.
        """
        if hasattr(player, "get_dominant_jersey_color_roi"):
            try:
                return player.get_dominant_jersey_color_roi(frame)
            except Exception:
                pass
        return player.get_dominant_jersey_color(frame)

    def _push_sample(self, tid: int, sample: tuple[int, int, int]) -> None:
        arr = self.jersey_samples_by_track.setdefault(tid, [])
        if len(arr) < self.samples_per_track:
            arr.append(sample)

    def _bgr_to_lab(self, bgr: np.ndarray) -> np.ndarray:
        """BGR -> LAB (float32)"""
        bgr_u8 = np.clip(bgr, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr_u8.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        return lab.astype(np.float32)

    def _fit_team_clusters(self) -> None:
        """
        Warmup sonunda tüm sample'lardan K=2 cluster öğren.

        Cluster -> team_id eşlemesi:
        - LAB'de chroma (a,b) düşük olan taraf genelde beyaz/açık forma olur (Team 2)
        - diğer cluster = renkli forma (Team 1)
        """
        all_samples: list[tuple[int, int, int]] = []
        for samples in self.jersey_samples_by_track.values():
            all_samples.extend(samples)

        if len(all_samples) < 30:
            # sample azsa KMeans güvenilir olmaz
            self.team_centers_lab = None
            self.team_cluster_to_id = {}
            return

        bgr_arr = np.array(all_samples, dtype=np.float32)
        lab_arr = self._bgr_to_lab(bgr_arr)

        data = lab_arr.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        K = 2
        attempts = 5
        _compactness, _labels, centers = cv2.kmeans(
            data, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
        )

        centers = centers.astype(np.float32)

        # "beyaz" cluster'ı bulmak için chroma (a,b) büyüklüğüne bakıyoruz
        ab = centers[:, 1:3] - 128.0
        chroma = np.sqrt(np.sum(ab * ab, axis=1))
        white_cluster = int(np.argmin(chroma))
        color_cluster = 1 - white_cluster

        self.team_centers_lab = centers
        self.team_cluster_to_id = {color_cluster: 1, white_cluster: 2}

    def _predict_team_from_color_kmeans(self, bgr: tuple[int, int, int]) -> int:
        """KMeans merkezleri varsa en yakın LAB merkeze göre takım tahmini."""
        if self.team_centers_lab is None or not self.team_cluster_to_id:
            return -1

        lab = self._bgr_to_lab(np.array([bgr], dtype=np.float32))[0]
        dists = np.sum((self.team_centers_lab - lab) ** 2, axis=1)
        cluster = int(np.argmin(dists))
        return int(self.team_cluster_to_id.get(cluster, 1))

    def _finalize_medians_and_threshold(self) -> None:
        """
        Warmup bitince:
        - her track için median BGR çıkar
        - KMeans çalışırsa takımı cluster'a göre ata ve kilitle
        - KMeans çalışmazsa R-B threshold fallback devreye girer
        """
        scores = []
        for tid, samples in self.jersey_samples_by_track.items():
            if not samples:
                continue
            data = np.array(samples, dtype=np.int32)
            med = np.median(data, axis=0)
            bgr = (int(med[0]), int(med[1]), int(med[2]))
            self.jersey_median_by_track[tid] = bgr
            scores.append(bgr[2] - bgr[0])  # fallback için R-B

        # KMeans ile cluster öğren
        self._fit_team_clusters()

        # Fallback threshold
        if self.team_centers_lab is None:
            if scores:
                self.score_threshold = float(np.median(np.array(scores, dtype=np.float32)))
            else:
                self.score_threshold = 0.0

        # Warmup sonrası hard-lock: bu noktada kim hangi takım -> sabitle
        for tid, bgr in self.jersey_median_by_track.items():
            pred = self._predict_team_from_color_kmeans(bgr)
            if pred == -1:
                thr = self.score_threshold or 0.0
                pred = 2 if (bgr[2] - bgr[0]) > thr else 1
            self.team_id_by_track[tid] = pred

        self._warmup_done = True

    # ----------------------------
    # Voting + locking (warmup sonrası veya stabil değilse yardımcı)
    # ----------------------------
    def _add_vote(self, tid: int, team_id: int) -> None:
        votes = self.team_votes_by_track.setdefault(tid, {1: 0, 2: 0})
        votes[team_id] += 1

        # kilitliyse hiç dokunma
        if tid in self.team_id_by_track:
            return

        total = votes[1] + votes[2]
        if total < self.lock_after_votes:
            return

        # bariz çoğunluk varsa kilitle
        if abs(votes[1] - votes[2]) >= self.lock_margin:
            self.team_id_by_track[tid] = 1 if votes[1] > votes[2] else 2

    def _predict_team_from_color(self, bgr: tuple[int, int, int]) -> int:
        # Önce KMeans
        pred = self._predict_team_from_color_kmeans(bgr)
        if pred != -1:
            return pred

        # Fallback: R-B threshold
        thr = self.score_threshold if self.score_threshold is not None else 0.0
        score = float(bgr[2]) - float(bgr[0])
        return 2 if score > thr else 1

    def _team_color_for_player(self, player: Player, frame: np.ndarray) -> tuple[int, int, int]:
        tid = int(player.track_id)

        # kilitli -> sabit renk
        if tid in self.team_id_by_track:
            return self.team_colors[self.team_id_by_track[tid]]

        # sample al
        sample = self._get_jersey_sample(player, frame)
        self._push_sample(tid, sample)

        # warmup sonrası yeni track geldi: kısa sample ile hemen kilitle
        if self._warmup_done:
            arr = self.jersey_samples_by_track.get(tid, [])
            if len(arr) >= self.new_track_lock_samples:
                data = np.array(arr, dtype=np.int32)
                med = np.median(data, axis=0)
                bgr_med = (int(med[0]), int(med[1]), int(med[2]))
                self.team_id_by_track[tid] = self._predict_team_from_color(bgr_med)
                return self.team_colors[self.team_id_by_track[tid]]

        # vote topla
        pred_team = self._predict_team_from_color(sample)
        self._add_vote(tid, pred_team)

        # kilitlendi mi?
        if tid in self.team_id_by_track:
            return self.team_colors[self.team_id_by_track[tid]]

        # kilitlenmediyse: anlık çoğunluğu göster (flicker azalsın)
        votes = self.team_votes_by_track.get(tid, {1: 0, 2: 0})
        maj_team = 1 if votes[1] >= votes[2] else 2
        return self.team_colors[maj_team]

    # ----------------------------
    # Drawing helpers
    # ----------------------------
    def composeFrame(self, big, small, pos=(0, 0), scale=0.25) -> np.ndarray:
        hsmall, wsmall = small.shape[:2]
        self.scaleBig = scale

        sh, sw = 226, 420
        small = cv2.resize(small, (sw, sh))
        self.scaleSmall = sw / wsmall

        x, y = pos
        big[y:y + sh, x:x + sw] = small
        self.picture_in_picture_section = (y, y + sh, x, x + sw)
        return big

    def drawPointsOnFrame(self, frame, points: np.ndarray = None) -> np.ndarray:
        annotated = frame.copy()
        if points is not None:
            annotated = self.point_drawer.drawPoints(annotated, points)
        return annotated

    def drawBoxOnFrame(self, frame, box: np.ndarray, color: str = "#00FF00", thickness: int = 2) -> np.ndarray:
        annotated = frame.copy()
        color_bgr = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        color_bgr = color_bgr[::-1]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, thickness)
        return annotated

    @staticmethod
    def draw_player_ellipse(frame: np.ndarray, player: Player, color: tuple[int, int, int], is_possessor: bool = False):
        x1, y1, x2, y2 = player.as_int_tuple()
        number = player.track_id if player.track_id is not None else 0
        cx, cy = int(player.center[0]), int(player.center[1])
        foot_x, foot_y = int(player.foot[0]), int(player.foot[1])

        w = max(12, x2 - x1)
        ax1 = int(w * 0.55)
        ax2 = int(w * 0.22)

        cv2.ellipse(frame, (cx, foot_y), (ax1, ax2), 0, 0, 360, (0, 0, 0), 5, cv2.LINE_AA)

        if is_possessor:
            cv2.ellipse(frame, (cx, foot_y), (ax1 + 10, ax2 + 8), 0, 0, 360, (0, 255, 255), 4, cv2.LINE_AA)

        cv2.ellipse(frame, (cx, foot_y), (ax1, ax2), 0, 0, 360, color, 3, cv2.LINE_AA)

        label = str(number)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lx = cx - tw // 2
        ly = max(th + 10, y1 - 10)

        cv2.rectangle(frame, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 5), color, 1)
        cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # ----------------------------
    # Main rendering
    # ----------------------------
    def drawAllFrames(
        self,
        frames: list[np.ndarray],
        small: np.ndarray,
        point_per_small: list[np.ndarray],
        points_per_frame: list[np.ndarray],
        players_per_frame: list[list[Player]] = None,
        tactical_players_per_frame: list[dict[int, list[float, float]]] = None,
        ball_per_frame: list[Ball] = None,
        out_path: str | None = None,
        fps: float = 30.0,
        codec: str = "mp4v",
        debug: bool = False,
    ):
        frames_out = []

        writer = None
        if out_path is not None:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frameImg = cv2.imread("images/basketball_court.png")
        frameImg = self.drawPointsOnFrame(frameImg, point_per_small)

        # Warmup: ilk N frame sadece sample topluyoruz
        warm_n = min(self.warmup_frames, len(frames) - 1)
        if (not self._warmup_done) and players_per_frame is not None and warm_n > 0:
            for idx in range(warm_n):
                plist = players_per_frame[idx]
                if not plist:
                    continue
                fr = frames[idx]
                for p in plist:
                    if p.track_id is None:
                        continue
                    tid = int(p.track_id)
                    if len(self.jersey_samples_by_track.get(tid, [])) >= self.samples_per_track:
                        continue
                    self._push_sample(tid, self._get_jersey_sample(p, fr))

            self._finalize_medians_and_threshold()

        # Asıl render
        for frame_idx, frame in enumerate(frames):
            if frame_idx % 50 == 0:
                print(f"Rendering {frame_idx}/{len(frames)}")

            frameTactical = frameImg.copy()
            frameSpec = frame.copy()

            if debug:
                h, w = frameSpec.shape[:2]
                cv2.putText(
                    frameSpec,
                    str(frame_idx),
                    (w - 200, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            frameSpec = self.drawPointsOnFrame(frameSpec, points_per_frame[frame_idx])

            if players_per_frame and players_per_frame[frame_idx] is not None:
                for player in players_per_frame[frame_idx]:
                    if player.track_id is None:
                        continue

                    color = self._team_color_for_player(player, frame)

                    # video üstündeki ellipse
                    is_possessor = (getattr(player, "class_id", None) == 99)
                    DrawWindow.draw_player_ellipse(frameSpec, player, color, is_possessor=is_possessor)

                    # minimap üstündeki nokta
                    if tactical_players_per_frame is not None:
                        tactical_data = tactical_players_per_frame[frame_idx]
                        if tactical_data and player.track_id in tactical_data:
                            coord = tactical_data[player.track_id]
                            frameTactical = self.point_drawer.drawSpecifiedPoint(
                                coord[0], coord[1], frameTactical, color=color, label=player.track_id
                            )

                if ball_per_frame is not None and ball_per_frame[frame_idx] is not None:
                    ball = ball_per_frame[frame_idx]
                    frameSpec = self.drawBoxOnFrame(frameSpec, ball.xyxy, color="#0000FF", thickness=2)

            frame_out = self.composeFrame(frameSpec, frameTactical, pos=(10, 10), scale=0.3)

            if writer is not None:
                writer.write(frame_out)
            else:
                frames_out.append(frame_out)

        if writer is not None:
            writer.release()
            return None

        return frames_out
