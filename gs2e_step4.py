import torch
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
import os
import imageio
import sys
from PIL import Image
from argparse import ArgumentParser

# Add the gaussian-splatting directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera
from arguments import ModelParams, PipelineParams

def main():
    # 1. Setup paths
    source_path = "tandt/train"
    model_path = "output/tandt_train"
    output_path = "rendered_trajectory"
    os.makedirs(output_path, exist_ok=True)

    # 2. Initialize Argument Parser
    parser = ArgumentParser(description="GS2E Trajectory Rendering")
    mp = ModelParams(parser, sentinel=True)
    pipe = PipelineParams(parser)
    parser.add_argument("--iteration", default=7000, type=int)
    
    args = parser.parse_args([])
    args.source_path = source_path
    args.model_path = model_path
    args.resolution = 1
    
    model_params = mp.extract(args)
    pipe_params = pipe.extract(args)

    # 3. Load the trained Gaussians
    gaussians = GaussianModel(3)
    ply_path = os.path.join(model_path, "point_cloud", "iteration_7000", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    print("Loaded 3D Gaussian model.")
    
    scene = Scene(model_params, gaussians, load_iteration=7000, shuffle=False)
    orig_cameras = scene.getTrainCameras()
    print(f"Loaded {len(orig_cameras)} original cameras.")

    # 4. Extract Poses (R, T) and Intrinsics
    ref_cam = orig_cameras[0]
    W, H = ref_cam.image_width, ref_cam.image_height
    FoVx, FoVy = ref_cam.FoVx, ref_cam.FoVy
    
    Rs = []
    Ts = []
    for cam in orig_cameras:
        w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
        Rs.append(w2c[:3, :3])
        Ts.append(w2c[:3, 3])
        
    Rs = np.array(Rs)
    Ts = np.array(Ts)
    N = len(Rs)

    # 5. Trajectory Smoothing (Eq 7, 8)
    w = 2 # window half-width
    Rs_smooth = []
    Ts_smooth = []
    for i in range(N):
        start = max(0, i - w)
        end = min(N, i + w + 1)
        Ts_smooth.append(np.mean(Ts[start:end], axis=0))
        mid = (start + end - 1) // 2
        Rs_smooth.append(Rs[mid])
        
    Rs_smooth = np.array(Rs_smooth)
    Ts_smooth = np.array(Ts_smooth)

    # 6. Calculate Displacements and Cumulative Path Length (Eq 9)
    alphas, betas = 1.0, 1.0
    deltas = []
    for i in range(N - 1):
        R1 = Rotation.from_matrix(Rs_smooth[i])
        R2 = Rotation.from_matrix(Rs_smooth[i+1])
        theta_rad = (R1.inv() * R2).magnitude() 
        dist = np.linalg.norm(Ts_smooth[i+1] - Ts_smooth[i])
        deltas.append(alphas * theta_rad + betas * dist)
        
    deltas = np.array(deltas)
    s = np.zeros(N)
    s[1:] = np.cumsum(deltas)
    total_path = s[-1]

    # 7. Velocity Profile and Target Path Lengths (Appendix A.1)
    gamma = 5.0 # interpolation multiplier
    M = int(gamma * N)
    
    t_j = np.linspace(0, 1, M)
    v_t = 0.25 * np.sin(t_j) + 1.1
    
    dt = 1.0 / (M - 1)
    cumulative_v = np.cumsum(v_t) * dt
    s_tilde = total_path * (cumulative_v / cumulative_v[-1])
    s_tilde[0] = 0

    # 8. Interpolate Trajectory (Cubic B-Spline)
    T_interp = np.array([CubicSpline(s, Ts_smooth[:, d])(s_tilde) for d in range(3)]).T
    
    quats = Rotation.from_matrix(Rs_smooth).as_quat()
    Q_interp = np.array([CubicSpline(s, quats[:, d])(s_tilde) for d in range(4)]).T
    Q_interp = Q_interp / np.linalg.norm(Q_interp, axis=1, keepdims=True)
    R_interp = Rotation.from_quat(Q_interp).as_matrix()

    print(f"Generated dense trajectory with {M} poses.")

    # 9. Render the Sequence
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    images = []
    
    # Create a dummy black PIL image to satisfy the Camera constructor
    dummy_image = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))

    print("Rendering high-FPS sequence (this may take a few minutes)...")
    for j in range(M):
        cam = Camera(
            resolution=(W, H),
            colmap_id=j,
            R=R_interp[j],
            T=T_interp[j],
            FoVx=FoVx,
            FoVy=FoVy,
            depth_params=None,
            image=dummy_image,
            invdepthmap=None,
            image_name=f"frame_{j:04d}",
            uid=j,
            trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
            data_device="cuda"
        )
        
        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipe_params, background)
            rgb = render_pkg["render"]
            
        rgb_np = (np.clip(rgb.cpu().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        images.append(rgb_np)
        
        if j % 100 == 0:
            print(f"Rendered frame {j}/{M}")

    # Save as video
    video_path = os.path.join(output_path, "trajectory.mp4")
    imageio.mimwrite(video_path, images, fps=60, quality=8)
    print(f"\n✅ SUCCESS! Saved smooth trajectory video to {video_path}")

if __name__ == "__main__":
    main()
