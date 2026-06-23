import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def main():
    video_path = "rendered_trajectory/trajectory.mp4"
    output_dir = "event_stream_output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading video from {video_path} using OpenCV...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: total_frames = 1505
        
    # Read first frame to get dimensions
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
        
    H, W = first_frame.shape[:2]
    print(f"Video resolution: {H}x{W}, FPS: {fps}, Total frames: {total_frames}")

    # DVS-Voltmeter Physics Parameters
    c = 1.0
    dt = 1.0 / fps
    sigma = 0.05

    print(f"Simulating DVS-Voltmeter events (Streaming mode to save RAM)...")

    voltage = np.zeros((H, W), dtype=np.float32)
    events = []
    prev_log_gray = None
    
    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # OpenCV reads in BGR, convert to grayscale directly
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        log_gray = np.log(gray / 255.0 + 1e-6)
        
        if prev_log_gray is not None:
            dL = log_gray - prev_log_gray
            noise = np.random.normal(0, sigma * np.sqrt(dt), (H, W)).astype(np.float32)
            voltage += dL + noise

            # ON events
            pos_mask = voltage >= c
            if np.any(pos_mask):
                ys, xs = np.where(pos_mask)
                ts = np.full(len(xs), frame_idx * dt, dtype=np.float32)
                ps = np.ones(len(xs), dtype=np.int8)
                events.append(np.column_stack([ts, xs, ys, ps]))
                voltage[pos_mask] -= c

            # OFF events
            neg_mask = voltage <= -c
            if np.any(neg_mask):
                ys, xs = np.where(neg_mask)
                ts = np.full(len(xs), frame_idx * dt, dtype=np.float32)
                ps = -np.ones(len(xs), dtype=np.int8)
                events.append(np.column_stack([ts, xs, ys, ps]))
                voltage[neg_mask] += c
                
        prev_log_gray = log_gray
        
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")
            
        frame_idx += 1
            
    cap.release()

    if len(events) == 0:
        print("No events generated.")
        return

    all_events = np.concatenate(events, axis=0)
    all_events = all_events[all_events[:, 0].argsort()]
    
    print(f"✅ Generated {len(all_events)} total events!")

    npz_path = os.path.join(output_dir, "events.npz")
    np.savez(npz_path, events=all_events)
    print(f"Saved raw events to {npz_path}")

    print("Generating event visualizations...")
    pos_events = all_events[all_events[:, 3] == 1]
    neg_events = all_events[all_events[:, 3] == -1]

    pos_img = np.zeros((H, W), dtype=np.float32)
    neg_img = np.zeros((H, W), dtype=np.float32)

    np.add.at(pos_img, (pos_events[:, 2].astype(int), pos_events[:, 1].astype(int)), 1)
    np.add.at(neg_img, (neg_events[:, 2].astype(int), neg_events[:, 1].astype(int)), 1)

    max_val = max(pos_img.max(), neg_img.max(), 1e-6)
    pos_img = np.clip(pos_img / max_val, 0, 1)
    neg_img = np.clip(neg_img / max_val, 0, 1)

    vis_img = np.zeros((H, W, 3), dtype=np.uint8)
    vis_img[:, :, 0] = (pos_img * 255).astype(np.uint8)
    vis_img[:, :, 2] = (neg_img * 255).astype(np.uint8)

    vis_path = os.path.join(output_dir, "event_visualization.png")
    plt.figure(figsize=(10, 6))
    plt.imshow(vis_img)
    plt.title(f"GS2E Event Stream (Red=ON, Blue=OFF)\nTotal Events: {len(all_events)}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {vis_path}")

if __name__ == "__main__":
    main()
