import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import gc

# ==== SuperPoint/SuperGlue Wrapper ====
sys.path.append(str(Path(__file__).parent / "SuperGluePretrainedNetwork"))
from models.superpoint import SuperPoint
from models.superglue import SuperGlue

print(f"torch version: {torch.__version__}")
print(f"CUDA available? {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

superpoint = SuperPoint({'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 1024}).eval().to(device)
superglue = SuperGlue({'weights': 'indoor'}).eval().to(device)

SHOW_MATCH_VIS = True  # 視覺化開關

def load_image_gray(path, resize_to_2k=True):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Image {path} not found!"
    img = img.astype(np.float32) / 255.0
    if resize_to_2k:
        target_w = 3840
        target_h = 2160
        h, w = img.shape[:2]
        # 等比例縮放，讓最短邊>=target對應邊
        scale = max(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # 中心裁切到 2560x1440
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        img = img[start_y:start_y + target_h, start_x:start_x + target_w]
    return img

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        x = x.float().to(device)
        assert x.device.type == device.type, f"Tensor not on device: {x.device}, target: {device}"
        return x
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().to(device)
        assert x.device.type == device.type, f"Tensor not on device: {x.device}, target: {device}"
        return x
    elif isinstance(x, (list, tuple)):
        if len(x) > 0 and isinstance(x[0], torch.Tensor):
            x = [t.cpu().detach().numpy() if isinstance(t, torch.Tensor) else t for t in x]
        x_np = np.array(x)
        x = torch.from_numpy(x_np).float().to(device)
        assert x.device.type == device.type, f"Tensor not on device: {x.device}, target: {device}"
        return x
    else:
        raise TypeError(f"Unknown type: {type(x)}")

def extract_superpoint_features(img_path, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)  # 確保 cache 目錄存在
    cache_path = os.path.join(cache_dir, Path(img_path).name + ".superpoint.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            feats = pickle.load(f)
        return feats
    img = load_image_gray(img_path, resize_to_2k=True)
    try:
        inp = torch.from_numpy(img)[None, None].to(device)
        pred = superpoint({'image': inp})
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[警告] CUDA OOM! 若能接受，請改用 CPU 執行。")
            raise e
        else:
            raise e
    feats = {
        'image': inp.cpu().numpy(),
        'keypoints': pred['keypoints'][0].detach().cpu().numpy(),
        'descriptors': pred['descriptors'][0].detach().cpu().numpy(),
        'scores': pred['scores'][0].detach().cpu().numpy()
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(feats, f)
    # 嚴格釋放 GPU memory
    del img
    del inp
    del pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feats

def auto_detect_max_batch(img_list, cache_dir, min_batch=1, max_batch=20):
    print("[INFO] 自動偵測單批最大可處理圖片數...")
    for batch_size in range(min_batch, max_batch + 1):
        try:
            tmp = []
            for img_path in img_list[:batch_size]:
                feats = extract_superpoint_features(img_path, cache_dir)
                tmp.append(feats)
                del feats
            tmp.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[INFO] 測試 batch_size={batch_size} 爆 VRAM，極限為 {batch_size-1}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return batch_size - 1 if batch_size > 1 else 1
            else:
                raise e
    print(f"[INFO] 最大測試 batch_size={max_batch} 未爆，極限為 {max_batch}")
    return max_batch

def cache_features_for_images(img_list, cache_dir, batch_size=1):
    os.makedirs(cache_dir, exist_ok=True)
    feats_dict = {}
    N = len(img_list)
    for start in tqdm(range(0, N, batch_size), desc=f"Extracting & caching SuperPoint features (batch_size={batch_size})"):
        for img_path in img_list[start:start+batch_size]:
            feats_dict[img_path.name] = extract_superpoint_features(img_path, cache_dir)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return feats_dict

def extract_and_match_from_feats(feat1, feat2):
    def to_tensor_local(arr):
        if isinstance(arr, torch.Tensor):
            return arr.to(device)
        return torch.from_numpy(arr).float().to(device)
    def ensure_batch_dim(x, dim=0):
        if isinstance(x, np.ndarray):
            x = to_tensor_local(x)
        if x.dim() == 2:
            return x.unsqueeze(dim)
        if x.dim() == 1:
            return x.unsqueeze(dim)
        return x
    def ensure_image_shape(img):
        img_t = to_tensor_local(img)
        if img_t.dim() == 2:
            img_t = img_t.unsqueeze(0).unsqueeze(0)
        elif img_t.dim() == 3:
            img_t = img_t.unsqueeze(0) if img_t.shape[0] != 1 else img_t
            if img_t.shape[1] != 1:
                img_t = img_t.unsqueeze(1)
        return img_t

    data = {
        'keypoints0': ensure_batch_dim(feat1['keypoints']),
        'descriptors0': ensure_batch_dim(feat1['descriptors']),
        'scores0': ensure_batch_dim(feat1['scores']),
        'keypoints1': ensure_batch_dim(feat2['keypoints']),
        'descriptors1': ensure_batch_dim(feat2['descriptors']),
        'scores1': ensure_batch_dim(feat2['scores']),
        'image0': ensure_image_shape(feat1['image']),
        'image1': ensure_image_shape(feat2['image']),
    }
    matches = superglue(data)
    matches0 = matches['matches0'][0].cpu().numpy()
    valid = matches0 > -1
    mkpts0 = data['keypoints0'][0][valid].cpu().numpy()
    mkpts1 = data['keypoints1'][0][matches0[valid]].cpu().numpy()
    del data
    del matches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mkpts0, mkpts1

def draw_matches_window(img1_path, img2_path, kpts1, kpts2, matches_count, name1, name2, window_name="Matches", max_matches=50):
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    max_width = 800
    scale1 = scale2 = 1.0
    if img1.shape[1] > max_width:
        scale1 = max_width / img1.shape[1]
        img1 = cv2.resize(img1, (0, 0), fx=scale1, fy=scale1)
        kpts1 = kpts1 * scale1
    if img2.shape[1] > max_width:
        scale2 = max_width / img2.shape[1]
        img2 = cv2.resize(img2, (0, 0), fx=scale2, fy=scale2)
        kpts2 = kpts2 * scale2

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    out_img = np.zeros((max(h1, h2)+60, w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img1
    out_img[:h2, w1:w1+w2] = img2

    num_matches = min(len(kpts1), max_matches)
    for i in range(num_matches):
        pt1 = tuple(np.round(kpts1[i]).astype(int))
        pt2 = tuple(np.round(kpts2[i]).astype(int))
        pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))
        color = tuple(np.random.randint(0,255,3).tolist())
        cv2.circle(out_img, pt1, 3, color, -1)
        cv2.circle(out_img, pt2_shifted, 3, color, -1)
        cv2.line(out_img, pt1, pt2_shifted, color, 1)

    text1 = f"{name1}"
    text2 = f"{name2}"
    match_info = f"Matches: {matches_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    cv2.putText(out_img, text1, (10, h1+30), font, font_scale, (0,255,0), font_thickness, cv2.LINE_AA)
    cv2.putText(out_img, text2, (w1+10, h2+30), font, font_scale, (0,255,0), font_thickness, cv2.LINE_AA)
    center_x = (w1 + w2)//2 - 80
    cv2.putText(out_img, match_info, (center_x, 40), font, 1.1, (0,0,255), 3, cv2.LINE_AA)
    cv2.imshow(window_name, out_img)
    cv2.waitKey(1)
    return True

def read_images_txt(path, file_list=None, debug_lines=20):
    images = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    idx = 0
    n = len(lines)
    image_count = 0
    while idx < n:
        line = lines[idx].strip()
        if line == '' or line[0] == '#':
            idx += 1
            continue
        elems = line.split()
        if len(elems) < 10:
            idx += 1
            continue
        image_id = int(elems[0])
        qvec = np.array([float(x) for x in elems[1:5]])
        tvec = np.array([float(x) for x in elems[5:8]])
        cam_id = int(elems[8])
        name = elems[9]
        idx += 1
        xys = []
        point3D_ids = []
        if idx < n:
            ptline = lines[idx].strip()
            pt_elems = ptline.split()
            for i in range(0, len(pt_elems), 3):
                try:
                    x = float(pt_elems[i])
                    y = float(pt_elems[i+1])
                    pid = int(float(pt_elems[i+2]))
                    xys.append([x, y])
                    point3D_ids.append(pid)
                except Exception as e:
                    print(f"[ERROR] image {name} 解析 2D點失敗: {e}")
                    break
            idx += 1
        images[name] = {
            'image_id': image_id,
            'qvec': qvec,
            'tvec': tvec,
            'cam_id': cam_id,
            'name': name,
            'xys': np.array(xys),
            'point3D_ids': np.array(point3D_ids)
        }
    if file_list is not None:
        file_names = set([p.name for p in file_list])
        images_keys = set(images.keys())
        diff = file_names - images_keys
        if diff:
            print(f"[WARNING] images.txt 缺少以下圖片 meta: {diff}")
        else:
            print("[CHECK] images.txt 與 images/ 目錄一致")
    return images

def read_points3D_txt(path):
    points3D = {}
    with open(path, 'r') as f:
        for line in f:
            if line[0] == '#' or line.strip() == '':
                continue
            elems = line.strip().split()
            point_id = int(elems[0])
            xyz = np.array([float(elems[1]), float(elems[2]), float(elems[3])])
            rgb = np.array([int(elems[4]), int(elems[5]), int(elems[6])])
            error = float(elems[7])
            track = [int(x) for x in elems[8:]]
            points3D[point_id] = {
                'xyz': xyz,
                'rgb': rgb,
                'error': error,
                'track': track
            }
    return points3D

def write_images_txt(images, path):
    with open(path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for name, img in images.items():
            f.write(f"{img['image_id']} {' '.join([str(x) for x in img['qvec']])} "
                    f"{' '.join([str(x) for x in img['tvec']])} {img['cam_id']} {img['name']}\n")
            for xy, pid in zip(img['xys'], img['point3D_ids']):
                f.write(f"{xy[0]} {xy[1]} {pid} ")
            f.write("\n")

def write_points3D_txt(points, path):
    with open(path, 'w') as f:
        f.write("# 3D point list\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pid, pt in points.items():
            xyz = pt['xyz']
            rgb = pt['rgb']
            error = pt['error']
            track = ' '.join(map(str, pt['track']))
            f.write(f"{pid} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} {error} {track}\n")

def get_all_images(img_dir):
    return sorted([p for p in Path(img_dir).glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])

def find_best_match_pairs(scene1_imgs, scene2_imgs, scene1_feats, scene2_feats):
    match_pairs = []
    for idx2, img2 in enumerate(tqdm(scene2_imgs, desc="Scene2 Images", dynamic_ncols=True)):
        best_num = 0
        best_img1 = None
        best_kpts1 = None
        best_kpts2 = None
        for idx1, img1 in enumerate(scene1_imgs):
            kpts1, kpts2 = extract_and_match_from_feats(scene1_feats[img1.name], scene2_feats[img2.name])
            if SHOW_MATCH_VIS:
                draw_matches_window(
                    img1, img2, kpts1, kpts2,
                    matches_count=len(kpts1),
                    name1=img1.name, name2=img2.name
                )
            if len(kpts1) > best_num:
                best_num = len(kpts1)
                best_img1 = img1
                best_kpts1 = kpts1
                best_kpts2 = kpts2
        tqdm.write(f"==> {img2.name} best matches {best_img1.name} ({best_num} matches)")
        if best_num > 20:
            match_pairs.append((best_img1, img2, best_kpts1, best_kpts2))
    cv2.destroyAllWindows()
    return match_pairs

def find_nearest_3d(xy, xys, point3D_ids, points3D):
    if len(xys) == 0: return None
    dists = np.linalg.norm(xys - xy, axis=1)
    idx = np.argmin(dists)
    pid = point3D_ids[idx]
    if pid == -1 or pid not in points3D: return None
    return points3D[pid]['xyz']

def get_3d_point_pairs(match_pairs, images1, points3D_1, images2, points3D_2):
    xyz1_list, xyz2_list = [], []
    for img1, img2, kpts1, kpts2 in match_pairs:
        name1, name2 = img1.name, img2.name
        if name1 not in images1:
            print(f"找不到 {name1} in images1, available keys: {list(images1.keys())[:5]} ...")
            continue
        if name2 not in images2:
            print(f"找不到 {name2} in images2, available keys: {list(images2.keys())[:5]} ...")
            continue
        xys1 = images1[name1]['xys']
        pids1 = images1[name1]['point3D_ids']
        xys2 = images2[name2]['xys']
        pids2 = images2[name2]['point3D_ids']
        for xy1, xy2 in zip(kpts1, kpts2):
            pt1 = find_nearest_3d(xy1, xys1, pids1, points3D_1)
            pt2 = find_nearest_3d(xy2, xys2, pids2, points3D_2)
            if pt1 is not None and pt2 is not None:
                xyz1_list.append(pt1)
                xyz2_list.append(pt2)
    return np.array(xyz1_list), np.array(xyz2_list)

def estimate_similarity_transform(src_pts, dst_pts):
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = np.sum(S) / np.trace(src_centered.T @ src_centered)
    t = dst_mean - scale * R @ src_mean
    return R, t, scale

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
    ])

def rotmat2qvec(R):
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr+1.0)*2
        q[0] = 0.25*s
        q[1] = (R[2,1] - R[1,2])/s
        q[2] = (R[0,2] - R[2,0])/s
        q[3] = (R[1,0] - R[0,1])/s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            q[0] = (R[2,1] - R[1,2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0,1] + R[1,0]) / s
            q[3] = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            q[0] = (R[0,2] - R[2,0]) / s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            q[0] = (R[1,0] - R[0,1]) / s
            q[1] = (R[0,2] + R[2,0]) / s
            q[2] = (R[1,2] + R[2,1]) / s
            q[3] = 0.25 * s
    return q

def transform_images(images, R, t, scale):
    new_images = {}
    for name, img in images.items():
        R0 = qvec2rotmat(img['qvec'])
        t0 = img['tvec']
        C = -R0.T @ t0
        C_new = scale*R @ C + t
        R_new = R @ R0
        t_new = -R_new @ C_new
        qvec_new = rotmat2qvec(R_new)
        new_images[name] = img.copy()
        new_images[name]['qvec'] = qvec_new
        new_images[name]['tvec'] = t_new
    return new_images

def transform_points3D(points, R, t, scale):
    new_points = {}
    for pid, pt in points.items():
        xyz = pt['xyz']
        xyz_new = scale*R @ xyz + t
        new_points[pid] = pt.copy()
        new_points[pid]['xyz'] = xyz_new
    return new_points

def transform_scene2_to_scene1(scene1_dir, scene2_dir, output_dir):
    scene1_imgs = get_all_images(os.path.join(scene1_dir, "images"))
    scene2_imgs = get_all_images(os.path.join(scene2_dir, "images"))

    images1 = read_images_txt(os.path.join(scene1_dir, "sparse", "images.txt"))
    images2 = read_images_txt(os.path.join(scene2_dir, "sparse", "images.txt"))

    points3D_1 = read_points3D_txt(os.path.join(scene1_dir, "sparse", "points3D.txt"))
    points3D_2 = read_points3D_txt(os.path.join(scene2_dir, "sparse", "points3D.txt"))

    print(f"[CHECK] scene1 images.txt 解析到 {len(images1)} 張")
    print(f"[CHECK] scene2 images.txt 解析到 {len(images2)} 張")

    # 1. 快取 scene1 特徵點
    scene1_cache = os.path.join(scene1_dir, "superpoint_cache")
    print("[INFO] 開始提取 scene1 特徵點 ...")
    if device.type == "cuda":
        max_batch1 = auto_detect_max_batch(scene1_imgs, scene1_cache, min_batch=1, max_batch=10)
    else:
        max_batch1 = 1
    scene1_feats = cache_features_for_images(scene1_imgs, scene1_cache, batch_size=max_batch1)
    del scene1_feats
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[INFO] scene1 特徵點提取完畢，VRAM 已清空。")

    # 3. 快取 scene2 特徵點
    scene2_cache = os.path.join(scene2_dir, "superpoint_cache")
    print("[INFO] 開始提取 scene2 特徵點 ...")
    if device.type == "cuda":
        max_batch2 = auto_detect_max_batch(scene2_imgs, scene2_cache, min_batch=1, max_batch=10)
    else:
        max_batch2 = 1
    scene2_feats = cache_features_for_images(scene2_imgs, scene2_cache, batch_size=max_batch2)
    print("[INFO] scene2 特徵點提取完畢。")

    print("[INFO] 重新讀取 scene1 特徵點 cache ...")
    scene1_feats = {}
    for img_path in scene1_imgs:
        cache_path = os.path.join(scene1_cache, img_path.name + ".superpoint.pkl")
        with open(cache_path, 'rb') as f:
            scene1_feats[img_path.name] = pickle.load(f)

    match_pairs = find_best_match_pairs(scene1_imgs, scene2_imgs, scene1_feats, scene2_feats)
    xyz1, xyz2 = get_3d_point_pairs(match_pairs, images1, points3D_1, images2, points3D_2)
    print(f"找到 {len(xyz1)} 組3D點對")
    if len(xyz1) < 5:
        print("匹配3D點對太少，請檢查圖片/配對結果")
        return
    R, t, scale = estimate_similarity_transform(xyz2, xyz1)
    images2_new = transform_images(images2, R, t, scale)
    points3D_2_new = transform_points3D(points3D_2, R, t, scale)
    os.makedirs(output_dir, exist_ok=True)
    write_images_txt(images2_new, os.path.join(output_dir, "images.txt"))
    write_points3D_txt(points3D_2_new, os.path.join(output_dir, "points3D.txt"))
    print("轉換完成，輸出到", output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python scene_alignment_allinone_auto_batch_2k_16_9.py scene1_dir scene2_dir output_dir")
        print("例子: python scene_alignment_allinone_auto_batch_2k_16_9.py scene1 scene2 scene2_reg_to_scene1")
        sys.exit(1)
    transform_scene2_to_scene1(sys.argv[1], sys.argv[2], sys.argv[3])