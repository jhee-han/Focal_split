import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

#1. Load two images from pkl dataset.
def load_images_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data[0], data[1]

#2 Change to green images
def convert_to_green(image1, image2):
    green1 = image1[:, :, 1]
    green2 = image2[:, :, 1]
    return green1, green2

#3 Aberration correction; apply averaging box filter
def aberration_correction(green_image, kernel_size=21):
    if green_image.dtype == np.uint8:
        green_image = green_image.astype(np.float32) / 255.0
    else:
        green_image = green_image.astype(np.float32)

    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2) #1/21*21
    convolutioned_img = cv2.filter2D(green_image, -1, kernel)
    filtered_out_background = green_image - convolutioned_img
    return filtered_out_background

#4 Noise attenuation; apply Gaussian filter
def noise_attenuation(filtered_image):
    smoothed_image = cv2.GaussianBlur(filtered_image, (11, 11),0)
    return smoothed_image

# SIFT nees uint8 images, so we need to convert float images to uint8
def prepare_sift_images(image1, image2):
    image1 = to_float01(image1)
    image2 = to_float01(image2)

    # RGB 입력이면 특징점 추출용 grayscale로 변환
    if image1.ndim == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    if image2.ndim == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    sift_image1 = np.clip(image1 * 255, 0, 255).astype(np.uint8)
    sift_image2 = np.clip(image2 * 255, 0, 255).astype(np.uint8)

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
def compute_derivatives(warped_image, image1):
    I_s_approx = 0.5 * (image1 - warped_image)

    image_mean = 0.5 * (warped_image + image1)

    # # first version of laplacian_I_approx
    # laplacian_I_approx = cv2.Laplacian(
    #     image_mean,
    #     cv2.CV_32F,
    #     ksize=1,
    # )
    # second version of laplacian_I_approx
    laplacian_I_approx = image_mean - cv2.GaussianBlur(image_mean, (11, 11), 0)

    return I_s_approx, laplacian_I_approx

#9 Depth estimation
def depth_estimation(
    S,
    L,
    A=1.23,
    B=0.19,
    patch_size=21,
):
    S = np.asarray(S, dtype=np.float32)
    L = np.asarray(L, dtype=np.float32)

    V = L
    W = A *L + B * S

    VW = cv2.boxFilter(V * W, -1, (patch_size, patch_size), normalize=False)
    WW = cv2.boxFilter(W * W, -1, (patch_size, patch_size), normalize=False)

    depth = VW/WW
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

    # min_depth, max_depth = np.percentile(valid_depth, [2, 98])
    # if min_depth == max_depth:
    #     max_depth = min_depth + 1e-8
    min_depth, max_depth = 0.4, 1.0

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


def to_float01(image):
    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0

    return image.astype(np.float32)


def display_comparison(
    dataset,
    our_img0,
    our_img1,
    our_aligned,
    our_depth,
    our_S,
    depth_range=(0.4, 1.0),
    derivative_limit=None,
    confidence_threshold=None,
    rotate180=True,
    save_path=None,
):
    """
    Top: author-provided arrays.
    Bottom: our inputs and computed outputs.

    confidence_threshold=None:
        Compare full-frame depth before confidence filtering.
    confidence_threshold=1e-5:
        Apply each row's S**2 mask to its depth, for display only.
    """
    author_depth = np.asarray(dataset[3], dtype=np.float32)
    author_S = np.asarray(dataset[4], dtype=np.float32)
    our_depth = np.asarray(our_depth, dtype=np.float32)
    our_S = np.asarray(our_S, dtype=np.float32)

    pairs = [
        (to_float01(dataset[0]), to_float01(our_img0)),
        (to_float01(dataset[1]), to_float01(our_img1)),
        (to_float01(dataset[2]), to_float01(our_aligned)),
        (author_depth, our_depth),
        (author_S, our_S),
    ]

    titles = [
        "Image 1",
        "Image 2",
        "Aligned Image 2",
        "Predicted depth",
        "Signed derivative S",
    ]

    for title, (author, ours) in zip(titles, pairs):
        if author.shape != ours.shape:
            raise ValueError(
                f"{title}: shape mismatch "
                f"{author.shape} vs {ours.shape}"
            )

    # One shared, symmetric range for both signed derivative maps.
    if derivative_limit is None:
        finite_values = np.concatenate([
            np.abs(author_S[np.isfinite(author_S)]),
            np.abs(our_S[np.isfinite(our_S)]),
        ])
        if finite_values.size == 0:
            raise ValueError("No finite derivative values")

        derivative_limit = max(float(finite_values.max()), 1e-8)

    if derivative_limit <= 0:
        raise ValueError("derivative_limit must be positive")

    depth_cmap = plt.get_cmap("jet").copy()
    depth_cmap.set_bad("white")
    depth_cmap.set_under("magenta")
    depth_cmap.set_over("black")

    derivative_cmap = plt.get_cmap("RdBu_r").copy()
    derivative_cmap.set_bad("lightgray")

    fig, axes = plt.subplots(
        2, 5,
        figsize=(23, 9),
        constrained_layout=True,
    )

    depth_artist = None
    derivative_artist = None

    for col, (author, ours) in enumerate(pairs):
        for row, array in enumerate((author, ours)):
            ax = axes[row, col]
            displayed = np.array(array, copy=True)

            if col == 3:
                mask = ~np.isfinite(displayed)

                if confidence_threshold is not None:
                    S = author_S if row == 0 else our_S
                    mask |= (
                        ~np.isfinite(S)
                        | (S**2 <= confidence_threshold)
                    )

                displayed = np.ma.array(displayed, mask=mask)

            elif col == 4:
                displayed = np.ma.masked_invalid(displayed)

            if rotate180:
                displayed = np.rot90(displayed, 2)

            if col < 3:
                # RGB images already share the [0, 1] display scale.
                ax.imshow(
                    np.clip(displayed, 0, 1),
                    interpolation="nearest",
                )

            elif col == 3:
                depth_artist = ax.imshow(
                    displayed,
                    cmap=depth_cmap,
                    vmin=depth_range[0],
                    vmax=depth_range[1],
                    interpolation="nearest",
                )

            else:
                derivative_artist = ax.imshow(
                    displayed,
                    cmap=derivative_cmap,
                    vmin=-derivative_limit,
                    vmax=derivative_limit,
                    interpolation="nearest",
                )

            prefix = "Author" if row == 0 else "Ours"
            ax.set_title(f"{prefix}: {titles[col]}")
            ax.axis("off")

    fig.colorbar(
        depth_artist,
        ax=axes[:, 3].tolist(),
        label="Depth (m)",
        extend="both",
        shrink=0.8,
    )
    fig.colorbar(
        derivative_artist,
        ax=axes[:, 4].tolist(),
        label="Signed derivative S (not confidence)",
        extend="both",
        shrink=0.8,
    )

    mode = (
        "Before confidence filtering"
        if confidence_threshold is None
        else f"Depth masked where S² ≤ {confidence_threshold:g}"
    )
    fig.suptitle(
        f"Focal Split comparison — {mode}",
        fontsize=16,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    plt.show()
    return fig

def main():
    pkl_path = 'CVPR.pkl'  
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)

    image1 = to_float01(dataset[0])
    image2 = to_float01(dataset[1])

    sift_image1, sift_image2 = prepare_sift_images(image1, image2)

    Homography = descriptor_matching_Homography(sift_image1, sift_image2)
    warped_image = warp_image(image2, Homography)

    green1, green2 = convert_to_green(image1, warped_image)
    filtered1 = aberration_correction(green1)
    filtered2 = aberration_correction(green2)

    denoised1 = noise_attenuation(filtered1)
    denoised2 = noise_attenuation(filtered2)

    I_s_approx, laplacian_I_approx = compute_derivatives(denoised2, denoised1)
    depth = depth_estimation(
        I_s_approx,
        laplacian_I_approx,
        A=1.23,
        B=0.19,
        patch_size=21
    )
    print(f"Valid depth pixels: {np.isfinite(depth).mean():.1%}")
    ''''''

    display_comparison(
        dataset=dataset,
        our_img0=image1,
        our_img1=image2,
        our_aligned=warped_image,
        our_depth=depth,
        our_S=I_s_approx,
        confidence_threshold=None,
        rotate180=True,
        save_path=(
            "comparison_all.png"
        ),
    )


if __name__ == "__main__":
    main()