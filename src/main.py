import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

file_path = "/home/user/Загрузки/Telegram Desktop/data_colors/data/Bo7TsgWtiiCCY6Esy2786.json"


def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Ошибка загрузки JSON: {e}")
        return None


def process_shot_data(data):
    trajectories = {i: [] for i in range(16)}
    for row in data:
        if isinstance(row, list) and len(row) == 16:
            for ball_id, values in enumerate(row):
                if isinstance(values, list) and len(values) >= 2:
                    x, y = values[0], values[1]
                    if x == -1 or y == -1:
                        trajectories[ball_id].append(None)
                    else:
                        trajectories[ball_id].append((x, y))
    return trajectories


def process_ticks(ticks):
    tol = 1e-8
    processed = []
    found_zero = False
    for tick in ticks:
        if abs(tick) < tol:
            if not found_zero:
                processed.append(0.0)
                found_zero = True
        elif tick > 0:
            processed.append(tick)
    return processed


def plot_trajectories(trajectories, title="Trajectories"):
    fig, ax = plt.subplots(figsize=(8, 6))
    for ball_id, points in trajectories.items():
        valid_points = [(p[0], p[1]) for p in points if p is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            ax.plot(x_vals, y_vals, marker="o", linestyle="-")
    ax.set_xlabel("X", labelpad=-10)
    ax.set_ylabel("Y", labelpad=-10, rotation=0)
    ax.xaxis.set_label_coords(1.05, 0)
    ax.yaxis.set_label_coords(0, 1.02)
    ax.set_title(title, pad=35)
    ax.grid()

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    xticks = process_ticks(ax.get_xticks())
    yticks = process_ticks(ax.get_yticks())
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    def y_formatter(y, pos):
        if abs(y) < 1e-8:
            return ""
        return f"{y:.1f}"

    ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))

    fig.canvas.draw()
    offset = 0.02
    for label in ax.get_xticklabels():
        if label.get_text() == "0.0":
            x, y = label.get_position()
            label.set_position((x - offset, y))
            label.set_ha("right")

    plt.setp(ax.get_yticklabels(), ha="right")
    plt.show()


class KalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise):
        self.dt = dt
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.Q = process_noise * np.eye(4)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.R = measurement_noise * np.eye(2)

    def initialize(self, x, y, vx=0, vy=0):
        self.x = np.array([[x], [y], [vx], [vy]])

    def predict(self):
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.x

    def update(self, measurement):
        z = np.array(measurement).reshape(2, 1)
        residual = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(residual)
        I = np.eye(self.P.shape[0])
        self.P = (I - K.dot(self.H)).dot(self.P)
        return self.x


def resolve_collision(pos1, vel1, pos2, vel2):
    delta_pos = pos1 - pos2
    dist = np.linalg.norm(delta_pos)
    if dist < 1e-8:
        return vel1, vel2
    n = delta_pos / dist
    p = 2 * np.dot(vel1 - vel2, n) / 2
    vel1_new = vel1 - p * n
    vel2_new = vel2 + p * n
    return vel1_new, vel2_new


def label_collision(pos1, vel1, pos2, vel2, angle_threshold=30):
    delta = pos2 - pos1
    norm_delta = np.linalg.norm(delta)
    if norm_delta < 1e-8:
        return "неопределено"
    n = delta / norm_delta
    norm_v1 = np.linalg.norm(vel1)
    norm_v2 = np.linalg.norm(vel2)
    if norm_v1 < 1e-3 or norm_v2 < 1e-3:
        return "неопределено"
    angle1 = np.degrees(np.arccos(np.clip(np.dot(vel1, n) / norm_v1, -1.0, 1.0)))
    angle2 = np.degrees(np.arccos(np.clip(np.dot(vel2, -n) / norm_v2, -1.0, 1.0)))
    if angle1 < angle_threshold and angle2 < angle_threshold:
        return "центральное"
    else:
        return "косой"


def filter_real_data(trajectories, dt=1, process_noise=1e-2, measurement_noise=1e-1, collision_radius=1.0):
    ball_ids = list(trajectories.keys())
    N = len(trajectories[ball_ids[0]])

    kf_dict = {}
    for b in ball_ids:
        first_point = None
        for p in trajectories[b]:
            if p is not None:
                first_point = p
                break
        if first_point is not None:
            kf = KalmanFilter(dt, process_noise, measurement_noise)
            kf.initialize(first_point[0], first_point[1], 0, 0)
            kf_dict[b] = kf

    filtered_trajectories = {b: [] for b in ball_ids}
    collision_events = []

    for frame in range(N):
        current_states = {}
        for b in ball_ids:
            if b not in kf_dict:
                filtered_trajectories[b].append(None)
                continue
            kf = kf_dict[b]
            kf.predict()
            meas = trajectories[b][frame]
            if meas is not None:
                kf.update(meas)
            current_states[b] = kf.x.copy()

        positions = {}
        velocities = {}
        for b, state in current_states.items():
            px, py, vx, vy = state.flatten()
            positions[b] = np.array([px, py])
            velocities[b] = np.array([vx, vy])

        ball_id_list = list(positions.keys())
        for i in range(len(ball_id_list)):
            for j in range(i + 1, len(ball_id_list)):
                id1 = ball_id_list[i]
                id2 = ball_id_list[j]
                if np.linalg.norm(positions[id1] - positions[id2]) < collision_radius * 2:
                    collision_type = label_collision(
                        positions[id1], velocities[id1],
                        positions[id2], velocities[id2]
                    )
                    collision_events.append({
                        "frame": frame,
                        "ball1": id1,
                        "ball2": id2,
                        "type": collision_type
                    })
                    v1_new, v2_new = resolve_collision(
                        positions[id1], velocities[id1],
                        positions[id2], velocities[id2]
                    )
                    velocities[id1] = v1_new
                    velocities[id2] = v2_new
                    kf_dict[id1].x[2, 0] = v1_new[0]
                    kf_dict[id1].x[3, 0] = v1_new[1]
                    kf_dict[id2].x[2, 0] = v2_new[0]
                    kf_dict[id2].x[3, 0] = v2_new[1]

        for b in ball_ids:
            if b not in kf_dict:
                filtered_trajectories[b].append(None)
            else:
                px = kf_dict[b].x[0, 0]
                py = kf_dict[b].x[1, 0]
                filtered_trajectories[b].append((px, py))

    return filtered_trajectories, collision_events, kf_dict


def predict_future(kf_dict, filtered_trajectories, start_frame, M, collision_radius=1.0):
    collision_events = []
    predicted_trajectories = {b: [] for b in filtered_trajectories.keys()}

    for frame in range(start_frame, start_frame + M):
        current_states = {}
        for b, kf in kf_dict.items():
            kf.predict()
            current_states[b] = kf.x.copy()

        positions = {}
        velocities = {}
        for b, state in current_states.items():
            px, py, vx, vy = state.flatten()
            positions[b] = np.array([px, py])
            velocities[b] = np.array([vx, vy])

        ball_id_list = list(positions.keys())
        for i in range(len(ball_id_list)):
            for j in range(i + 1, len(ball_id_list)):
                id1 = ball_id_list[i]
                id2 = ball_id_list[j]
                if np.linalg.norm(positions[id1] - positions[id2]) < collision_radius * 2:
                    collision_type = label_collision(
                        positions[id1], velocities[id1],
                        positions[id2], velocities[id2]
                    )
                    collision_events.append({
                        "frame": frame,
                        "ball1": id1,
                        "ball2": id2,
                        "type": collision_type
                    })
                    v1_new, v2_new = resolve_collision(
                        positions[id1], velocities[id1],
                        positions[id2], velocities[id2]
                    )
                    velocities[id1] = v1_new
                    velocities[id2] = v2_new
                    kf_dict[id1].x[2, 0] = v1_new[0]
                    kf_dict[id1].x[3, 0] = v1_new[1]
                    kf_dict[id2].x[2, 0] = v2_new[0]
                    kf_dict[id2].x[3, 0] = v2_new[1]

        for b in predicted_trajectories.keys():
            if b not in kf_dict:
                predicted_trajectories[b].append(None)
            else:
                px = kf_dict[b].x[0, 0]
                py = kf_dict[b].x[1, 0]
                predicted_trajectories[b].append((px, py))

    return predicted_trajectories, collision_events


def plot_two_stages(observed, predicted, title1="Filtered (0..N-1)", title2="Predicted (N..N+M-1)"):
    plot_trajectories(observed, title=title1)
    plot_trajectories(predicted, title=title2)


def prepare_unigine_data(trajectories_dict):
    any_ball = next(iter(trajectories_dict.values()))
    M = len(any_ball)
    frames = []
    for i in range(M):
        frame_data = {"balls": []}
        for b in sorted(trajectories_dict.keys()):
            pos = trajectories_dict[b][i]
            if pos is None:
                frame_data["balls"].append({"x": -1, "y": -1})
            else:
                frame_data["balls"].append({"x": pos[0], "y": pos[1]})
        frames.append(frame_data)
    return {"frames": frames}


if __name__ == "__main__":
    data = load_json(file_path)
    if data:
        N_observed = 300
        M_predicted = 10
        observed_data = data[:N_observed]
        trajectories = process_shot_data(observed_data)
        filtered_real, collisions_real, kf_dict = filter_real_data(
            trajectories, dt=1, process_noise=1e-2, measurement_noise=1e-1, collision_radius=1.0
        )
        print("События столкновений (реальные кадры):")
        for event in collisions_real:
            print(f"[Frame {event['frame']}] Шары {event['ball1']} и {event['ball2']} => {event['type']}")
        predicted_future, collisions_future = predict_future(
            kf_dict, filtered_real, start_frame=N_observed, M=M_predicted, collision_radius=1.0
        )
        print("\nСобытия столкновений (прогнозные кадры):")
        for event in collisions_future:
            print(f"[Frame {event['frame']}] Шары {event['ball1']} и {event['ball2']} => {event['type']}")
        plot_two_stages(filtered_real, predicted_future,
                        title1=f"Фильтрация (0..{N_observed - 1})",
                        title2=f"Прогноз ({N_observed}..{N_observed + M_predicted - 1})")
        unigine_data_pred = prepare_unigine_data(predicted_future)
        with open("predicted_unigine.json", "w") as f:
            json.dump(unigine_data_pred, f, indent=2, ensure_ascii=False)
        print("\nПрогнозные данные для UNIGINE сохранены в 'predicted_unigine.json'.")
    else:
        print("Не удалось загрузить данные.")
