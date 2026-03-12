import numpy as np
from pathlib import Path  
import cv2
from skimage import measure
import matplotlib.pyplot as plt
import os 
import pandas as pd
import csv
from scipy import stats

def load_images_from_folder(folder_path):
    images = {}
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg']
    for file_path in folder_path.iterdir():
        if file_path.suffix.lower() in image_extensions:
            img = cv2.imread(str(file_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[file_path.name] = img_rgb
    return images
    
def get_image_size_from_df(name, df) :
    image_size = {}
    for _, row in df.iterrows():
        key = f"{name}_{row.iloc[3]}"
        size = (row.iloc[11], row.iloc[12])
        image_size[key] = size
    return image_size
    
def resize_images(images, size_map):
    resized_images = {}
    for name, img in images.items():
        if name in size_map:
            target_height, target_width = size_map[name]
            resized_images[name] = cv2.resize(img, (target_width, target_height))
    return resized_images
    
def merge_tiles_to_image(tiles, image_size_map, tile_h, tile_w):
    stitched_images = {}
    for base_name, (full_h, full_w) in image_size_map.items():
        print(f"Stitching image: {base_name}, target size: {full_h}x{full_w}")
        num_channels = 3 
        first_tile = next(iter(tiles.values()), None)
        if first_tile is not None:
            if first_tile.ndim == 3:
                num_channels = first_tile.shape[2]
            else: 
                num_channels = 1
        if num_channels > 1:
            stitched_image = np.zeros((full_h, full_w, num_channels), dtype=np.uint8)
        else:
            stitched_image = np.zeros((full_h, full_w), dtype=np.uint8)
        num_y = int(np.ceil(full_h / tile_h))
        num_x = int(np.ceil(full_w / tile_w))
        for j in range(num_y):
            for i in range(num_x):
                tile_name = f"{base_name}_{j}_{i}.png"
                tile = tiles.get(tile_name)
                if tile is None:
                    print(f"Warning: Tile {tile_name} not found! Area will be black.")
                    continue 
                y_start = j * tile_h
                x_start = i * tile_w
            
                h_to_copy = min(tile.shape[0], full_h - y_start)
                w_to_copy = min(tile.shape[1], full_w - x_start)
        
                stitched_image[y_start : y_start + h_to_copy, x_start : x_start + w_to_copy] = \
                    tile[-h_to_copy:, -w_to_copy:]
        
        stitched_images[base_name] = stitched_image
        
    return stitched_images
    
def calculate_precipitate_sizes(image_name, img,excluded_magnification,lower_bound_color,upper_bound_color,min_precipitate_pixel_area,nm_per_pixel_base_mag):
    try:
        magnification = int(image_name.split('X')[0].split('-')[1])
    except (IndexError, ValueError):
        print(f"Warning: Could not parse magnification from '{image_name}'. Skipping size calculation.")
        return []

    if magnification == excluded_magnification:
        return []

 
    mask = cv2.inRange(img, lower_bound_color, upper_bound_color)
    
 
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    
    pixel_areas = [prop.area for prop in props if prop.area >= min_precipitate_pixel_area]
    
 
    conversion_factor = (nm_per_pixel_base_mag / magnification)**2
    physical_areas = [area * conversion_factor * 10**6 for area in pixel_areas]
    
    return physical_areas


def analyze_carbide_shapes(image_path: Path,contour_area_range) -> tuple[list, list]:
    # Load image in BGR format
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Warning: Could not read image file: {image_path}")
        return [], []
 
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of all distinct shapes in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    aspect_ratios = []
    bounding_boxes = []
    min_area, max_area = contour_area_range

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area < area < max_area):
            continue
 
        rect = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(rect)
        box = np.int0(box_points)
 
        width, height = rect[1]
        if width <= 0 or height <= 0:
            continue
 
        aspect_ratio = max(width, height) / min(width, height)
        aspect_ratios.append(aspect_ratio)
        bounding_boxes.append(box)

    return aspect_ratios, bounding_boxes
    
    
def create_highlighted_image(original_image_path: Path, boxes: list,box_color_bgr) -> np.ndarray:
    img_bgr = cv2.imread(str(original_image_path))
    cv2.drawContours(img_bgr, boxes, -1, box_color_bgr, 2)
    return img_bgr

def process_steel_type(steel_name: str, image_folder: Path,output_folder,excluded_magnification,contour_area_range,box_color_bgr) -> list:
    print(f"\nProcessing steel type: {steel_name}...")
    image_folder=Path(image_folder)
    all_ratios = []

    # Prepare the output directory for this steel type's highlighted images
    output_image_dir = Path(os.path.join(output_folder,steel_name.replace(" ", "_")))

    output_image_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in image_folder.glob("*.png") if int(p.name.split('-')[-1].split('_')[0].split('X')[0])  != excluded_magnification]

    for image_path in image_paths:
        ratios, boxes = analyze_carbide_shapes(image_path,contour_area_range)
        
        if ratios:
            all_ratios.extend(ratios)
            
            # Create and save the highlighted image for visual verification
            highlighted_img = create_highlighted_image(image_path, boxes,box_color_bgr)
            save_path = output_image_dir / f"{image_path.stem}_highlighted.png"
            cv2.imwrite(str(save_path), highlighted_img)


    return all_ratios
    
def bootstrap_ci(data, statistic_func, n_bootstrap=2000, confidence_level=0.95):
    data = np.array(data)
 
    if len(data) == 0:
        return np.nan, (np.nan, np.nan)
        
    bootstrap_stats = []
    n_size = len(data)
    for _ in range(n_bootstrap):
        resample_indices = np.random.choice(n_size, size=n_size, replace=True)
        resample = data[resample_indices]
        stat = statistic_func(resample)
        bootstrap_stats.append(stat)
    point_estimate = statistic_func(data)
    alpha = (1 - confidence_level) / 2.0
    lower_bound = np.percentile(bootstrap_stats, alpha * 100)
    upper_bound = np.percentile(bootstrap_stats, (1 - alpha) * 100)
    
    return point_estimate, (lower_bound, upper_bound)



def plot_size_distribution(merged_size_list, output_dir, total_areas_lb, total_areas_tm):

    data_lb = merged_size_list['N5325BAINITE']
    data_tm = merged_size_list['N5440MARTENSITE']

    mean_lb, mean_ci_lb = bootstrap_ci(data_lb, np.mean)
    median_lb, median_ci_lb = bootstrap_ci(data_lb, np.median)
    mean_tm, mean_ci_tm = bootstrap_ci(data_tm, np.mean)
    median_tm, median_ci_tm = bootstrap_ci(data_tm, np.median)


    bins = np.logspace(3, 9, 30)
    counts_lb, _ = np.histogram(data_lb, bins=bins)
    counts_tm, _ = np.histogram(data_tm, bins=bins)
    counts_lb = counts_lb / total_areas_lb
    counts_tm = counts_tm / total_areas_tm
    global_max_y = max(np.max(counts_lb), np.max(counts_tm))


    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 25
    plt.rcParams['mathtext.fontset'] = 'stix' 

    color_lb, color_tm = '#779de2', '#f28be5'
    color_mean, color_median = 'green', 'red'


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    plot_configs = [
        {'ax': ax1, 'data': data_lb, 'mean': mean_lb, 'mean_ci': mean_ci_lb, 
         'median': median_lb, 'median_ci': median_ci_lb, 'color': color_lb, 'label': '(a) LB'},
        {'ax': ax2, 'data': data_tm, 'mean': mean_tm, 'mean_ci': mean_ci_tm, 
         'median': median_tm, 'median_ci': median_ci_tm, 'color': color_tm, 'label': '(b) TM'}
    ]

    for config in plot_configs:
        ax = config['ax']
        total_area = total_areas_lb if config['label'] == '(a) LB' else total_areas_tm

        ax.hist(config['data'], bins=bins, weights=np.ones_like(config['data']) / total_area,color=config['color'], 
                edgecolor='white', linewidth=1.2, alpha=0.5)

        ax.set_xscale('log')
        ax.set_ylim(0, global_max_y * 1.35)
        ax.set_xlabel('Carbide Size (nm²)')
        
        if ax == ax1:
            ax.set_ylabel('Number of carbides')

        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

        m_val, m_ci = config['mean'], config['mean_ci']
        md_val, md_ci = config['median'], config['median_ci']
        

        ax.axvline(m_val, color=color_mean, linestyle='--', linewidth=2, zorder=3)
        ax.axvline(md_val, color=color_median, linestyle='-.', linewidth=2, zorder=3)
        ax.axvspan(m_ci[0], m_ci[1], color=color_mean, alpha=0.2, zorder=1)
        ax.axvspan(md_ci[0], md_ci[1], color=color_median, alpha=0.2, zorder=1)


        m_coeff = m_val / 1000
        m_ci_l = m_ci[0] / 1000
        m_ci_h = m_ci[1] / 1000


        md_coeff = md_val / 1000
        md_ci_l = md_ci[0] / 1000
        md_ci_h = md_ci[1] / 1000


        mean_str = (f"Mean = {m_coeff:.2f} $\\times$ 10$^{{-3}}$\n"
                    f"95% CI: [{m_ci_l:.2f}, {m_ci_h:.2f}] $\\times$ 10$^{{-3}}$ $\mu$m$^2$")
        
        median_str = (f"Median = {md_coeff:.2f} $\\times$ 10$^{{-3}}$\n"
                      f"95% CI: [{md_ci_l:.2f}, {md_ci_h:.2f}] $\\times$ 10$^{{-3}}$ $\mu$m$^2$")


        y_ref = ax.get_ylim()[1]
        ax.text(m_val * 1.8, y_ref * 0.82, mean_str, color=color_mean, fontsize=25)
        ax.text(m_val * 1.8, y_ref * 0.64, median_str, color=color_median, fontsize=25)


        ax.text(0.95, 0.95, config['label'], transform=ax.transAxes,
                fontsize=30, va='top', ha='right')

    plt.tight_layout(pad=3.0)
    size_fig_path = os.path.join(output_dir, "carbide_size_distribution_final.png")
    plt.savefig(size_fig_path, dpi=300)
    print(f"Figure saved to: {size_fig_path}")
    plt.show()
    



def aspect_ratio_plot(aspect_ratio_data,output_dir):
    data_lb = aspect_ratio_data['N5325BAINITE']
    data_tm = aspect_ratio_data['N5440MARTENSITE']


    font_size = 16

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
 
    axes[0, 0].scatter(np.arange(len(data_lb)), data_lb, 
                       marker='^', s=30, facecolors='none', edgecolors='royalblue', alpha=0.7)
    axes[0, 0].set_ylabel("Carbide Aspect Ratio", fontsize=font_size)
    axes[0, 0].text(0.05, 0.95, '(a) LB', transform=axes[0, 0].transAxes, 
                    fontsize=font_size, va='top', ha='left')
 
    axes[1, 0].scatter(np.arange(len(data_tm)), data_tm, 
                       marker='o', s=30, facecolors='none', edgecolors='hotpink', alpha=0.3)
    axes[1, 0].set_xlabel("Carbide Index", fontsize=font_size)
    axes[1, 0].set_ylabel("Carbide Aspect Ratio", fontsize=font_size)
    axes[1, 0].text(0.05, 0.95, '(b) TM', transform=axes[1, 0].transAxes,
                    fontsize=font_size, va='top', ha='left')
 

    axes[0, 1].hist(data_lb, bins=np.linspace(0, 6, 20), color='royalblue',
                    edgecolor='gray', hatch='//', alpha=0.7)
    axes[0, 1].set_ylabel("Number of carbides", fontsize=font_size)
    axes[0, 1].text(0.95, 0.95, '(c) LB', transform=axes[0, 1].transAxes,
                    fontsize=font_size, va='top', ha='right')



    mean_lb, mean_ci_lb = bootstrap_ci(data_lb, np.mean)
    median_lb, median_ci_lb = bootstrap_ci(data_lb, np.median)

    mean_tm, mean_ci_tm = bootstrap_ci(data_tm, np.mean)
    median_tm, median_ci_tm = bootstrap_ci(data_tm, np.median)
 
    axes[0, 1].axvline(mean_lb, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_lb:.2f}')
    axes[0, 1].axvspan(mean_ci_lb[0], mean_ci_lb[1], color='red', alpha=0.2, label=f'Mean 95% CI: [{np.round(mean_ci_lb[0],2)}, {np.round(mean_ci_lb[1],2)}]')
   
    axes[0, 1].axvline(median_lb, color='green', linestyle=':', linewidth=2, label=f'Median: {median_lb:.2f}')
    axes[0, 1].axvspan(median_ci_lb[0], median_ci_lb[1], color='green', alpha=0.2, label=f'Median 95% CI: [{np.round(median_ci_lb[0],2)}, {np.round(median_ci_lb[1],2)}]')

    axes[0, 1].legend(fontsize=13)  

  

    axes[1, 1].hist(data_tm, bins=np.linspace(0, 6, 20), color='hotpink',
                    edgecolor='gray', hatch='//', alpha=0.3)
    axes[1, 1].set_xlabel("Aspect ratio", fontsize=font_size)
    axes[1, 1].set_ylabel("Number of carbides", fontsize=font_size)
    axes[1, 1].text(0.95, 0.95, '(d) TM', transform=axes[1, 1].transAxes,
                    fontsize=font_size, va='top', ha='right')

    axes[1, 1].axvline(mean_tm, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_tm:.2f}')
    axes[1, 1].axvspan(mean_ci_tm[0], mean_ci_tm[1], color='red', alpha=0.2, label=f'Mean 95% CI: [{np.round(mean_ci_tm[0],2)}, {np.round(mean_ci_tm[1],2)}]')
 
    axes[1, 1].axvline(median_tm, color='green', linestyle=':', linewidth=2, label=f'Median: {median_tm:.2f}')
    axes[1, 1].axvspan(median_ci_tm[0], median_ci_tm[1], color='green', alpha=0.2, label=f'Median 95% CI: [{np.round(median_ci_tm[0],2)}, {np.round(median_ci_tm[1],2)}]')

    axes[1, 1].legend(fontsize=13)  
 
    for ax in [axes[0, 0], axes[1, 0]]:
        ax.set_ylim(0, 15)

    for ax in [axes[0, 1], axes[1, 1]]:
        ax.set_xlim(1, 6)

    for ax in [axes[0, 0], axes[1, 0]]:
        ax.set_xlim(0, max(len(data_lb),len(data_tm)))

    size_fig_path=os.path.join(output_dir,"carbide_aspect_ratio_distribution_with_CI.png")
    print(f"The figure has been saved to {size_fig_path}")
    plt.savefig(size_fig_path, dpi=300)
    plt.show()
    
    
def print_sizes_aspect_ratios_to_csv(merged_size_list,aspect_ratio_data,output_dir,model_name):
    
    output_df = pd.DataFrame({name: pd.Series(sizes) for name, sizes in merged_size_list.items()})
    output_df.to_csv(output_dir /f"Size_Distribution_{model_name}.csv", index=False)
    print("\nAnalysis complete. Results of sizes have been saved to CSV and PNG files.")
      
    output_df = pd.DataFrame({name: pd.Series(sizes) for name, sizes in aspect_ratio_data.items()})
    output_df.to_csv(output_dir / f"Aspect_Ratio_Distribution_{model_name}.csv", index=False)
    print("\nAnalysis complete. Results of aspect ratios have been saved to CSV and PNG files.")
    
    with open(os.path.join(output_dir,f'Size_and_Aspect_ratio_p_values{model_name}.csv'), "w", newline='') as csv_file:
        writer = csv.writer(csv_file)

        t_stat, p_val = stats.ttest_ind(aspect_ratio_data['N5325BAINITE'], aspect_ratio_data['N5440MARTENSITE'])

        writer.writerow([f"t-statistic for carbide aspect ratios: {t_stat}"]) 
        writer.writerow([f"p-value for carbide aspect ratios: {p_val}"])     

        if p_val < 0.05:
            writer.writerow(["Conclusion: Reject the null hypothesis. The difference between the group means is statistically significant."]) 
        else:
            writer.writerow(["Conclusion: Fail to reject the null hypothesis. There is no statistically significant difference between the groups."]) 


        writer.writerow([]) 

        t_stat, p_val = stats.ttest_ind(merged_size_list['N5325BAINITE'], merged_size_list['N5440MARTENSITE'])

        writer.writerow([f"t-statistic for carbide sizes: {t_stat}"]) 
        writer.writerow([f"p-value for carbide sizes: {p_val}"])    

        if p_val < 0.05:
            writer.writerow(["Conclusion: Reject the null hypothesis. The difference between the group means is statistically significant."]) 
        else:
            writer.writerow(["Conclusion: Fail to reject the null hypothesis. There is no statistically significant difference between the groups."]) 

        print(f"File f'Size_and_Aspect_ratio_p_values{model_name}.csv' has been written successfully.")


