import cv2
import numpy as np
import os
from pathlib import Path

class Camera:
    def __init__(self, name = ""):
        self.name = name

        # Camera matrix intriniscs
        self.fx = 0
        self.fy = 0
        self.cx = 0
        self.cy = 0

        # Lens distorion intrinsics
        self._K = np.zeros((3,3), dtype=np.float32)
        self._dist = np.zeros((), dtype=np.float32)
        pass
    
    @property
    def K(self):
        K = np.array([[self.fx,0,cx],[0,self.fy,self.cy],[0,0,1]],dtype=np.float32)
        
        return K

def stitch_to_360(image_paths, orientations, K, dist_coeffs, 
                  output_size=(4096,2048)):
    """
    Given a list of image file paths (covering full sphere) and their orientations 
    (azimuth, zenith) in degrees, plus camera intrinsics K and distortion coeffs,
    stitch them into a single equirectangular panorama image of size output_size.
    
    Args:
        image_paths: list of strings, length N
        orientations: list of tuples (azimuth_deg, zenith_deg), length N
        K: 3x3 numpy array (camera intrinsics)
        dist_coeffs: distortion coefficients array (e.g. [k1,k2,p1,p2,k3])
        output_size: (width, height) of output panorama, e.g. (4096,2048)
    Returns:
        pano: numpy array (H x W x 3) stitched panorama
    """
    # Step 1: Load and undistort all images
    imgs = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        # undistort
        img_und = cv2.undistort(img, K, dist_coeffs)
        imgs.append(img_und)
    # Step 2: For each image project onto spherical coordinates and map to equirectangular
    W, H = output_size
    pano = np.zeros((H, W, 3), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    
    for index, (img, (az_deg, zen_deg)) in enumerate(zip(imgs, orientations)):
        h, w = img.shape[:2]
        # convert each pixel in pano to direction vector, then map to image
        # Compute spherical coordinate grid for output
        # Theta (azimuth) = x / W * 360°  (0 to 360)
        # Phi (zenith or elevation) = y / H * 180°  (0 = up, 180 = down).v  
        xv, yv = np.meshgrid(np.arange(W), np.arange(H))
        theta = (xv.astype(np.float32) / W) * 2*np.pi # 0 to 2π
        phi   = (yv.astype(np.float32) / H) * np.pi   # 0 to π

        # Direction vector in camera/world frame for this image orientation
        # We assume orientation gives rotation from camera‐forward to world direction
        # Convert az/zen to a rotation matrix R_world_cam
        az = np.deg2rad(az_deg)
        ze = np.deg2rad(zen_deg)


        # Here we assume: zen=0 => looking up (north pole), zen=90 => horizon, zen=180 => down
        # You may need to adjust based on your coordinate convention
        # First rotate around horizontal axis (X) by (zen-90°) then around vertical (Y) by az
        Rx = cv2.Rodrigues(np.array([1,0,0]) * (ze - np.pi/2))[0]
        Ry = cv2.Rodrigues(np.array([0,1,0]) * az)[0]
        R = Ry.dot(Rx)
        # R = Rx.dot(Ry)
        # Convert spherical output direction to vector
        vx = np.sin(phi) * np.cos(theta)
        vy = np.sin(phi) * np.sin(theta)
        vz = np.cos(phi)
        dirs = np.stack((vx, vy, vz), axis=-1)  # H x W x 3

        # Map dirs into camera coordinate of this shot: dir_cam = R^T * dirs (if R maps cam->world)
        dir_cam = dirs.reshape(-1,3).dot(R.T)
        x_cam = dir_cam[:,0] / dir_cam[:,2]
        y_cam = dir_cam[:,1] / dir_cam[:,2]

        # Project to pixel coordinates: u = fx * x_cam + cx ; v = fy * y_cam + cy
        fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
        u = fx * x_cam + cx
        v = fy * y_cam + cy

        u = u.reshape(H, W).astype(np.float32)
        v = v.reshape(H, W).astype(np.float32)

        # For points where dir_cam.z <= 0 (behind the camera) skip
        valid = (dir_cam.reshape(H, W,3)[:,:,2] > 0)

        # Sample image with cv2.remap
        map_x = u
        map_y = v
        sampled = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = valid.astype(np.float32)

        pano += sampled * mask[:,:,None]
        weight += mask

        print(f"Finnish processing image {index+1} out of {len(imgs)}", end="\r")

    # Normalize by weight
    # weight3 = weight[:,:,None]
    # pano = np.where(weight3>0, pano/weight3, pano)
    pano = np.clip(pano, 0, 255).astype(np.uint8)
    return pano

def merge_hdr(pano_images, exposure_times, output_path="hdr_output.hdr"):
    """
    Given a list of equirectangular panoramas (same orientation) at different exposure times,
    merge into an HDR image and save.
    
    Args:
        pano_images: list of file paths (or numpy arrays) in corresponding exposure order
        exposure_times: list of floats (in seconds) corresponding to each image
        output_path: filepath to save the HDR (.hdr or .exr)
    Returns:
        hdr: numpy array, float32 HDR image
    """
    # Load images
    imgs = [cv2.imread(p, cv2.IMREAD_COLOR).astype(np.float32)/255.0 for p in pano_images]
    # Convert exposure_times to numpy
    times = np.array(exposure_times, dtype=np.float32)
    # Use OpenCV's mergeDebevec or mergeRobertson
    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(imgs, times=times.copy())
    # tonemap for preview (optional)
    tonemap = cv2.createTonemapDrago(1.0,0.7)
    ldr = tonemap.process(hdr)
    ldr8 = np.clip(ldr*255,0,255).astype(np.uint8)
    cv2.imwrite("preview.jpg", ldr8)
    # Save HDR
    cv2.imwrite(output_path, hdr)
    return hdr

if __name__ == "__main__":
    # Example usage:
    # Suppose you have images named like “az_0_zen_30_exp0.jpg”, etc.
    hdr_name = "HDRI Tyholt" 
    exposures = [1/250, 1/30, 1/4]  # example shutter times in seconds
    exposures_names = ["1.250", "1.30", "1.4"]

    exposures = [1/30]
    exposures_names = ["1.30"]

    # define camera intrinsics and distortion
    img_w, img_h = 4032, 2268
    f_mm = 5.0                 # focal length from EXIF
    sensor_w_mm = 7.6          # approximate sensor width (mm)
    sensor_h_mm = 5.7          # approximate sensor height (mm)

    # focal lengths in pixels
    fx = f_mm / sensor_w_mm * img_w
    fy = f_mm / sensor_h_mm * img_h

    # principal point (assumed center)
    cx = img_w / 2.0
    cy = img_h / 2.0

    # radial / tangential distortion (unknown from EXIF)
    k1, k2, k3 = 0.0, 0.0, 0.0
    p1, p2 = 0.0, 0.0

    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    # Merge images into 360 photo
    for i in range(len(exposures)):
        folder = Path(f"./input/{hdr_name}/exp_{exposures_names[i]}/")
        img_paths = list(folder.glob("*.jpg"))


        # img_paths = [f"./input/{hdr_name}/exp_{exposures_names[i]}/az_000_zen_000.jpg", f"./input/{hdr_name}/exp_{exposures_names[i]}/az_000_zen_030.jpg"]

        # print(img_paths)

        orientations = [] # (az, zen) evry 30 azimuth from 000 to 330 
        for p in img_paths:
            # parse orientation from filename
            # e.g. az_030_zen_060.jpg
            fname = os.path.basename(p)
            fname = fname[0:-4] # remove .jpg from the file-name
            parts = fname.split('_')
            # print(parts)

            az = float(parts[1])
            zen = float(90) - float(parts[3]) # convert from 90 to -90 -> 0 to 180
            orientations.append((az, zen))
        
        # camera = Camera("Galaxy S21")
        pano = stitch_to_360(img_paths, orientations, K, dist, output_size=(4096,2048))
        
        folder = f"./output/{hdr_name}"
        os.makedirs(folder, exist_ok=True)        # ensure folder exists

        cv2.imwrite(f"{folder}/pano_exp{i}.jpg", pano)
        cv2.imshow("Pano", pano)
    
    # Then for HDR:
    # hdr = merge_hdr(["pano_exp0.jpg","pano_exp1.jpg","pano_exp2.jpg","pano_exp3.jpg","pano_exp4.jpg"], exposures, output_path=f"./output/{hdr_name}/{hdr_name}.hdr")
