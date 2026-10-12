import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

#1. Load two images from pkl dataset.
def load_images_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data[0], data[1]

#2 Change to grayscale images
def convert_to_grayscale(image):
    image1, image2 = load_images_from_pkl(image)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    return gray1, gray2

#3 Aberration correction; apply averaging box filter
def aberration_correction(gray_image, kernel_size=21):
    if gray_image.dtype == np.uint8:
        gray_image = gray_image.astype(np.float32) / 255.0
    else:
        gray_image = gray_image.astype(np.float32)

    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2) #1/21*21
    convolutioned_img = cv2.filter2D(gray_image, -1, kernel)
    filtered_out_background = gray_image - convolutioned_img
    return filtered_out_background

#4 Noise attenuation; apply Gaussian filter
def noise_attenuation(filtered_image):
    smoothed_image = cv2.GaussianBlur(filtered_image, (11, 11),0)
    return smoothed_image

# Sift needs images to be in the range [0, 255] and of type uint8. This function normalizes the input images to meet these requirements.
def prepare_sift_images(image1, image2):
    scale = np.percentile(
    np.abs(np.concatenate((image1.ravel(), image2.ravel()))),
    99,
    )
    scale = max(float(scale), 1e-8)

    sift_image1 = np.clip(127.5 + image1 * (127.5 / scale), 0, 255).astype(np.uint8)
    sift_image2 = np.clip(127.5 + image2 * (127.5 / scale), 0, 255).astype(np.uint8)
    return sift_image1, sift_image2

#5 SIFT feature extraction; apply SIFT algorithm
def sift_feature_extraction(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

#6 Descriptor matching and Homography estimation; apply RANSAC algorithm
def descriptor_matching_Homography(image1, image2, ratio=0.75):
    keypoints1, descriptors1 = sift_feature_extraction(image1)
    keypoints2, descriptors2 = sift_feature_extraction(image2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    Homography, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
    return Homography

#7 Image warping; apply Homography to warp the image
def warp_image(image, Homography):
    height, width = image.shape[:2]
    warped_image = cv2.warpPerspective(image, Homography, (width, height))
    return warped_image

#8 Two images derivates
def compute_derivatives(warped_image, image2):
    I_s_approx = warped_image - image2

    image_mean = 0.5 * (warped_image + image2)

    laplacian_I_approx = cv2.Laplacian(
        image_mean,
        cv2.CV_32F,
        ksize=1,
    )    
    return I_s_approx, laplacian_I_approx

#9 Depth estimation
def depth_estimation(
    I_s_approx,
    laplacian_I_approx,
    a,
    b,
    patch_size=21,
    confidence_threshold=0,#4e-5
):
    S = I_s_approx.astype(np.float32)
    D = laplacian_I_approx.astype(np.float32)
    term = b * D + S

    numerator = cv2.boxFilter(
        D * term,
        cv2.CV_32F,
        (patch_size, patch_size),
        normalize=True,
    )
    denominator = cv2.boxFilter(
        term * term,
        cv2.CV_32F,
        (patch_size, patch_size),
        normalize=True,
    )

    depth = np.full_like(numerator, np.nan)
    confidence = S * S
    valid = (
        np.isfinite(S)
        & np.isfinite(D)
        & np.isfinite(denominator)
        & (denominator > 1e-12)
        & (confidence >= confidence_threshold)
    )
    depth[valid] = a * numerator[valid] / denominator[valid]
    return depth

# Display two images
def display_images(image1, image2):
    
    cv2.imshow('Image 1', image1)
    cv2.imshow('Image 2', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

# 99th percentile normalization and display residual images(How to visualize the images are different)
def display_residual_images(image1, image2):
    scale = np.percentile(
        np.abs(np.concatenate((image1.ravel(), image2.ravel()))),
        99,
    )
    scale = max(scale, 1e-8)

    display1 = np.clip(0.5 + image1 / (2 * scale), 0, 1)
    display2 = np.clip(0.5 + image2 / (2 * scale), 0, 1)
    display1 = (display1 * 255).astype(np.uint8)
    display2 = (display2 * 255).astype(np.uint8)

    cv2.imshow('Image 1 - residual', display1)
    cv2.imshow('Image 2 - residual', display2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_depth(depth):
    valid_depth = depth[np.isfinite(depth)]
    if valid_depth.size == 0:
        raise ValueError("Depth map contains no finite values to display")

    min_depth, max_depth = np.percentile(valid_depth, [2, 98])
    if min_depth == max_depth:
        max_depth = min_depth + 1e-8

    depth_image = np.ma.masked_invalid(depth)
    figure, axis = plt.subplots(figsize=(8, 6))
    displayed = axis.imshow(
        np.flipud(np.fliplr(depth_image)),
        cmap="rainbow",
        vmin=min_depth,
        vmax=max_depth,
    )
    axis.set_title("Estimated depth")
    axis.axis("off")
    figure.colorbar(displayed, ax=axis, label="Depth value (display range)")
    plt.tight_layout()
    plt.show()

def display_confidence(I_s_approx):
    confidence = np.asarray(I_s_approx, dtype=np.float32) ** 2
    finite_confidence = confidence[np.isfinite(confidence)]

    if finite_confidence.size == 0:
        raise ValueError("Confidence map contains no finite values to display")

    vmin, vmax = np.percentile(finite_confidence, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1e-12

    figure, axis = plt.subplots(figsize=(8, 6))
    displayed = axis.imshow(
        np.flipud(np.fliplr(np.ma.masked_invalid(confidence))),
        cmap="gray_r",  # 높은 confidence를 검정색으로 표시
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title("Confidence map (S²)")
    axis.axis("off")
    figure.colorbar(displayed, ax=axis, label="Confidence (display range)")
    plt.tight_layout()
    plt.show()

def main():
    pkl_path = 'CVPR.pkl'  
    gray1, gray2 = convert_to_grayscale(pkl_path)
    filtered1 = aberration_correction(gray1)
    filtered2 = aberration_correction(gray2)
    denoised1 = noise_attenuation(filtered1)
    denoised2 = noise_attenuation(filtered2)
    sift_image1, sift_image2 = prepare_sift_images(denoised1, denoised2)
    Homography = descriptor_matching_Homography(sift_image1, sift_image2)
    warped_image = warp_image(denoised2, Homography)
    I_s_approx, laplacian_I_approx = compute_derivatives(warped_image, denoised1)
    depth = depth_estimation(
        I_s_approx,
        laplacian_I_approx,
        5.263,
        6.474,
        patch_size=21,
        confidence_threshold=0,#4e-5
    )
    print(f"Valid depth pixels: {np.isfinite(depth).mean():.1%}")
    ''''''
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)

    cvpr_aligned = dataset[2]
    our_aligned_rgb = cv2.warpPerspective(
        dataset[1],
        Homography,
        (gray1.shape[1], gray1.shape[0]),
    )

    def prepare_rgb_for_display(image):
        image = np.asarray(image)
        if image.dtype != np.uint8 and np.nanmax(image) > 1.0:
            image = image.astype(np.float32) / 255.0
        return np.clip(image, 0.0, 1.0) if image.dtype != np.uint8 else image

    residual_limit = np.percentile(np.abs(warped_image), 99)
    residual_limit = max(float(residual_limit), 1e-8)

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(prepare_rgb_for_display(cvpr_aligned))
    axes[0].set_title("Dataset aligned image (data[2])")

    axes[1].imshow(prepare_rgb_for_display(our_aligned_rgb))
    axes[1].set_title("Our Homography on original image2")

    axes[2].imshow(
        warped_image,
        cmap="gray",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    axes[2].set_title("Our warped denoised image (used for depth)")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()

    reference_depth = np.asarray(dataset[3], dtype=np.float32)
    reference_confidence = np.asarray(dataset[4], dtype=np.float32)
    estimated_confidence = np.asarray(I_s_approx, dtype=np.float32) ** 2

    if reference_depth.shape != depth.shape:
        raise ValueError(
            f"Depth shape mismatch: reference {reference_depth.shape}, "
            f"estimated {depth.shape}"
        )

    if reference_confidence.shape != estimated_confidence.shape:
        raise ValueError(
            f"Confidence shape mismatch: reference {reference_confidence.shape}, "
            f"estimated {estimated_confidence.shape}"
        )

    # 두 map 모두 현재 depth 표시와 같은 180도 방향으로 맞춤
    reference_depth = np.flipud(np.fliplr(reference_depth))
    estimated_depth = np.flipud(np.fliplr(depth))
    reference_confidence = np.flipud(np.fliplr(reference_confidence))
    estimated_confidence = np.flipud(np.fliplr(estimated_confidence))

    # Depth 비교: 동일한 색상 범위를 사용
    finite_depths = np.concatenate(
        (
            reference_depth[np.isfinite(reference_depth)],
            estimated_depth[np.isfinite(estimated_depth)],
        )
    )
    if finite_depths.size == 0:
        raise ValueError("No finite depth values available for comparison")

    depth_min, depth_max = np.percentile(finite_depths, [2, 98])
    if depth_max <= depth_min:
        depth_max = depth_min + 1e-8

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, depth_map, title in zip(
        axes,
        (reference_depth, estimated_depth),
        ("CVPR reference depth", "My estimated depth"),
    ):
        image = axis.imshow(
            np.ma.masked_invalid(depth_map),
            cmap="rainbow",
            vmin=depth_min,
            vmax=depth_max,
        )
        axis.set_title(title)
        axis.axis("off")

    figure.colorbar(image, ax=axes, label="Depth")
    plt.tight_layout()
    plt.show()

    # Confidence 비교: 각 map의 값 범위가 다를 수 있어 패널별로 정규화
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, confidence_map, title in zip(
        axes,
        (reference_confidence, estimated_confidence),
        ("CVPR reference confidence", "My confidence (S²)"),
    ):
        finite_confidence = confidence_map[np.isfinite(confidence_map)]
        if finite_confidence.size == 0:
            raise ValueError(f"{title} contains no finite values")

        conf_min, conf_max = np.percentile(finite_confidence, [2, 98])
        if conf_max <= conf_min:
            conf_max = conf_min + 1e-12

        image = axis.imshow(
            np.ma.masked_invalid(confidence_map),
            cmap="gray_r",
            vmin=conf_min,
            vmax=conf_max,
        )
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(image, ax=axis, label="Confidence (display range)")

    plt.tight_layout()
    plt.show()


    display_depth(depth)
    display_confidence(I_s_approx)
if __name__ == "__main__":
    main()