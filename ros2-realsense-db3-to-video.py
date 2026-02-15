from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import cv2
import numpy as np
import subprocess

# Configuration
BAGPATH = Path('data/rosbag2_2026_02_12-14_48_38_0.db3')
OUTPUT_DIR = Path('output')
ENABLE_COMPRESSION = False
BITRATE = '24000k'

# Topics
COLOR_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/depth/image_rect_raw'
IMU_TOPIC = '/imu_raw/Imu'


def get_typestore_foxy():
    return get_typestore(Stores.ROS2_FOXY)


def extract_color_frames(reader):
    color_frames = []
    fps = 30
    frame_width = None
    frame_height = None

    color_connections = [x for x in reader.connections if x.topic == COLOR_TOPIC]

    for connection, _, rawdata in reader.messages(connections=color_connections):
        msg = reader.deserialize(rawdata, connection.msgtype)

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        color_frames.append(frame)

        if frame_width is None:
            frame_height, frame_width = frame.shape[:2]

    return color_frames, fps, frame_width, frame_height


def extract_depth_frames(reader):
    depth_frames = []

    depth_connections = [x for x in reader.connections if x.topic == DEPTH_TOPIC]

    for connection, _, rawdata in reader.messages(connections=depth_connections):
        msg = reader.deserialize(rawdata, connection.msgtype)

        depth_frame = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

        depth_normalized = cv2.normalize(
            depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        depth_frames.append(depth_colored)

    return depth_frames


def extract_imu_data(reader):
    imu_connections = [x for x in reader.connections if x.topic == IMU_TOPIC]

    for connection, timestamp, rawdata in reader.messages(connections=imu_connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        print(msg.header.frame_id)


def create_video(frames, output_path, fps, width, height, enable_compression):
    if not frames:
        print(f"No frames to write for {output_path}")
        return

    codec = 'libx264' if enable_compression else 'mpeg4'
    crf = '23' if enable_compression else '0'

    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pixel_format', 'bgr24',
        '-video_size', f'{width}x{height}',
        '-framerate', str(fps),
        '-i', 'pipe:',
        '-c:v', codec,
        '-crf', crf,
        '-preset', 'medium'
    ]

    if enable_compression:
        cmd.extend(['-b:v', BITRATE])

    cmd.append(str(output_path))

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for frame in frames:
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    stderr = process.stderr.read().decode()
    process.wait()

    if process.returncode != 0:
        print(f"Error creating video: {stderr}")
    else:
        print(f"Video saved to {output_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    typestore = get_typestore_foxy()

    with AnyReader([BAGPATH], default_typestore=typestore) as reader:
        print("Extracting IMU data...")
        extract_imu_data(reader)

        print("Extracting color frames...")
        color_frames, fps, width, height = extract_color_frames(reader)

        print("Extracting depth frames...")
        depth_frames = extract_depth_frames(reader)

    if width is None or height is None:
        print("Error: Could not determine frame size.")
        return

    if color_frames:
        create_video(
            color_frames,
            OUTPUT_DIR / 'color_video.mp4',
            fps,
            width,
            height,
            ENABLE_COMPRESSION
        )

    if depth_frames:
        create_video(
            depth_frames,
            OUTPUT_DIR / 'depth_video.mp4',
            fps,
            width,
            height,
            ENABLE_COMPRESSION
        )


if __name__ == '__main__':
    main()