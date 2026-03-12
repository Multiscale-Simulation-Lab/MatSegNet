import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.gridspec as gridspec
import numpy as np
import os
import cv2
import copy
import scipy
import csv
import imageio
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']


def highlight_carbides(image_path, output_dir, number=0, name='Name', ratio=2):

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    comparison_highlighted_dir=os.path.join(output_dir,"Merged Highlighted Carbides")
    orientation_histogram_dir=os.path.join(output_dir,"Carbide Orientation")
    highlighted_dir=os.path.join(output_dir,"Carbide Highlighted")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(comparison_highlighted_dir, exist_ok=True)
    os.makedirs(orientation_histogram_dir, exist_ok=True)
    os.makedirs(highlighted_dir, exist_ok=True)
    

    img_raw = cv2.imread(image_path)
    if img_raw is None:
        print(f"Error: File not found or could not be opened at path: {image_path}")
        return None


    img_with_boxes = copy.deepcopy(img_raw)
    gray = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2GRAY)


    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    angle_list = []
    for c in contours:

        area = cv2.contourArea(c)
        if not (30 < area < 100000):
            continue
        

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = box.astype(int)


        width, height = rect[1]
        angle = rect[2]

        if width < height:
            angle = 90 - angle
        else:
            angle = -angle
            
        angle_list.append(angle)
        cv2.drawContours(img_with_boxes, [box], 0, (80, 127, 255), 2)


    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    axs[0].imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image', fontsize=20)
    axs[0].axis('off')
    
    axs[1].imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Carbide Orientation Highlighted', fontsize=20)
    axs[1].axis('off')

    comparison_filename = f"{name}_{number}_comparison.jpg"
    plt.savefig(os.path.join(comparison_highlighted_dir, comparison_filename), dpi=100, bbox_inches='tight')
    plt.close(fig)


    highlighted_filename = f"{name}_{number}_highlighted.jpg"
    cv2.imwrite(os.path.join(highlighted_dir  , highlighted_filename), img_with_boxes)


    if not angle_list:
        print(f"Warning: No valid carbides found in image {name}_{number}.")
        return 0.0


    fig, ax = plt.subplots(figsize=(16, 9))
    counts, bins, patches = ax.hist(angle_list, color='blue', alpha=0.7, bins=18, range=(-90, 90), edgecolor='black', linewidth=2)
    
    for rect_patch in patches:
        height = rect_patch.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        xy=(rect_patch.get_x() + rect_patch.get_width() / 2, height),
                        xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')
    
    ax.set_aspect(ratio)
    ax.set_title('Carbide Angle Distribution', fontsize=20)
    ax.set_xlim(-100, 100)
    ax.set_xlabel('Angle (°)', fontsize=20)
    ax.set_ylabel('Number of Carbides', fontsize=20)
    ax.set_xticks(np.arange(-90, 91, 10))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in', length=4)
    
    histogram_filename = f"{name}_{number}_histogram.jpg"
    plt.savefig(os.path.join(orientation_histogram_dir, histogram_filename), dpi=100)
    plt.close(fig)
    max_count = np.max(counts)
    total_count = np.sum(counts)
    angle_percentage = max_count / total_count if total_count > 0 else 0.0

    return angle_percentage
    
    
def get_percentage_list(dataset ,images_saved_dir,name='Name'):
    output_list = []
    for i, image_path in enumerate(dataset):
        print('Processing image {}: {}'.format(i, image_path))
        percentage = highlight_carbides(image_path, images_saved_dir,number=i, name=name)
        if percentage is not None:
            output_list.append(percentage)
        else:
            print(f"Skipping image {i} ({image_path}) due to an error.")

    return output_list


def get_area_percentage(dataset, color_channel=1, threshold=0.9):

    ratio_list = []
    for i, image_path in enumerate(dataset):
        try:
            img = imageio.imread(image_path)
            if len(img.shape) < 3 or img.shape[2] <= color_channel:
                print(f"Warning: Skipping image {i} ({image_path}) because it's grayscale or has too few channels.")
                continue 
            pixel_count = np.sum(img[:, :, color_channel] / 255 > threshold)
            total_pixels = img.shape[0] * img.shape[1]
            ratio = pixel_count / total_pixels if total_pixels > 0 else 0.0
            ratio_list.append(ratio)
        except FileNotFoundError:
            print(f"Error: File not found at {image_path}. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {image_path}: {e}. Skipping.")

    return ratio_list
    




def plot_comparison_figure(lb_data_a, tm_data_a, lb_data_b, tm_data_b, 
                           ylabel_a='Carbide Volume Fraction', ylabel_b='k', 
                           save_path=None):
    

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.labelweight'] = 'normal'

    def calculate_ci(data, confidence=0.95):
        """计算给定数据的均值的置信区间"""
        n = len(data)
        mean = np.mean(data)
        sem = scipy.stats.sem(data)
        if sem > 0:
            ci_range = scipy.stats.t.interval(confidence, n - 1, loc=mean, scale=sem)
            return mean, ci_range
        else:
            return mean, (mean, mean)


    lb_mean_a, lb_ci_a = calculate_ci(lb_data_a)
    tm_mean_a, tm_ci_a = calculate_ci(tm_data_a)
    lb_mean_b, lb_ci_b = calculate_ci(lb_data_b)
    tm_mean_b, tm_ci_b = calculate_ci(tm_data_b)


    combined_data_a = np.concatenate((lb_data_a, tm_data_a))
    
    if combined_data_a.size == 0:
        print("Warning: combined_data_a is empty, cannot generate plot.")
        return  
        
    min_a, max_a = np.min(combined_data_a), np.max(combined_data_a)
    range_a = max_a - min_a
    scatter_ylim_a = (min_a - 0.1 * range_a, max_a + 0.3 * range_a)
    hist_xlim_a = (min_a - 0.05 * range_a, max_a + 0.05 * range_a)
    counts_a_lb, _ = np.histogram(lb_data_a, bins=20)
    counts_a_tm, _ = np.histogram(tm_data_a, bins=20)
    max_count_a = max(counts_a_lb.max(), counts_a_tm.max())
    hist_ylim_a = (0, max_count_a * 1.3)

    combined_data_b = np.concatenate((lb_data_b, tm_data_b))
    min_b, max_b = combined_data_b.min(), combined_data_b.max()
    range_b = max_b - min_b
    scatter_ylim_b = (min_b - 0.1 * range_b, max_b + 0.3 * range_b)
    hist_xlim_b = (min_b - 0.05 * range_b, max_b + 0.05 * range_b)
    counts_b_lb, _ = np.histogram(lb_data_b, bins=25)
    counts_b_tm, _ = np.histogram(tm_data_b, bins=25)
    max_count_b = max(counts_b_lb.max(), counts_b_tm.max())
    hist_ylim_b = (0, max_count_b * 1.3)


    fig = plt.figure(figsize=(12, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3, height_ratios=[2, 1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])


    ax1.scatter(range(len(lb_data_a)), lb_data_a, facecolors='none', edgecolors='royalblue', marker='o', label='LB')
    ax1.scatter(range(len(tm_data_a)), tm_data_a, facecolors='none', edgecolors='orchid', marker='^', label='TM')
    ax1.axhline(y=lb_mean_a, color='royalblue', linestyle='-')
    ax1.axhline(y=tm_mean_a, color='orchid', linestyle='--')
    ax1.fill_between(range(len(lb_data_a)), lb_ci_a[0], lb_ci_a[1], color='royalblue', alpha=0.2)
    ax1.fill_between(range(len(lb_data_a)), tm_ci_a[0], tm_ci_a[1], color='orchid', alpha=0.2)
    ax1.set_title('(a)', loc='left', fontsize=16)
    ax1.set_xlabel('Image Index', fontsize=14)
    ax1.set_ylabel(ylabel_a, fontsize=14)
    ax1.text(0.05, 0.95, f'LB: Avg={lb_mean_a:.2f}, STD={np.std(lb_data_a):.2f}\n    95% CI: [{lb_ci_a[0]:.2f}, {lb_ci_a[1]:.2f}]\nTM: Avg={tm_mean_a:.2f}, STD={np.std(tm_data_a):.2f}\n    95% CI: [{tm_ci_a[0]:.2f}, {tm_ci_a[1]:.2f}]',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(loc='upper right', fontsize=12)


    ax2.scatter(range(len(lb_data_b)), lb_data_b, facecolors='none', edgecolors='royalblue', marker='o', label='LB')
    ax2.scatter(range(len(tm_data_b)), tm_data_b, facecolors='none', edgecolors='orchid', marker='^', label='TM')
    ax2.axhline(y=lb_mean_b, color='royalblue', linestyle='-')
    ax2.axhline(y=tm_mean_b, color='orchid', linestyle='--')
    ax2.fill_between(range(len(lb_data_b)), lb_ci_b[0], lb_ci_b[1], color='royalblue', alpha=0.2)
    ax2.fill_between(range(len(lb_data_b)), tm_ci_b[0], tm_ci_b[1], color='orchid', alpha=0.2)
    ax2.set_title('(b)', loc='left', fontsize=16)
    ax2.set_xlabel('Image Index', fontsize=14)
    ax2.set_ylabel(ylabel_b, fontsize=14)
    ax2.text(0.05, 0.95, f'LB: Avg={lb_mean_b:.2f}, STD={np.std(lb_data_b):.2f}\n    95% CI: [{lb_ci_b[0]:.2f}, {lb_ci_b[1]:.2f}]\nTM: Avg={tm_mean_b:.2f}, STD={np.std(tm_data_b):.2f}\n    95% CI: [{tm_ci_b[0]:.2f}, {tm_ci_b[1]:.2f}]',
             transform=ax2.transAxes, fontsize=12, verticalalignment='top')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.legend(loc='upper right', fontsize=12)


    common_bins_a = np.linspace(hist_xlim_a[0], hist_xlim_a[1], 20 + 1) 
    common_bins_b = np.linspace(hist_xlim_b[0], hist_xlim_b[1], 25 + 1) 
    

    ax3.hist(lb_data_a, bins=common_bins_a, color='royalblue', alpha=0.7, label='LB', edgecolor='black', linewidth=1.2)
    ax3.set_title('(c)', loc='left', fontsize=16)
    ax3.set_ylabel('Image number', fontsize=14)
    ax3.legend(fontsize=12)
    

    ax5.hist(tm_data_a, bins=common_bins_a, color='orchid', alpha=0.7, label='TM', edgecolor='black', linewidth=1.2)
    ax5.set_title('(e)', loc='left', fontsize=16)
    ax5.set_xlabel(ylabel_a.lower(), fontsize=14)
    ax5.set_ylabel('Image number', fontsize=14)
    ax5.legend(fontsize=12)
    

    ax4.hist(lb_data_b, bins=common_bins_b, color='royalblue', alpha=0.7, label='LB', edgecolor='black', linewidth=1.2)
    ax4.set_title('(d)', loc='left', fontsize=16)
    ax4.set_ylabel('Image number', fontsize=14)
    ax4.legend(fontsize=12)


    ax6.hist(tm_data_b, bins=common_bins_b, color='orchid', alpha=0.7, label='TM', edgecolor='black', linewidth=1.2)
    ax6.set_title('(f)', loc='left', fontsize=16)
    ax6.set_xlabel(ylabel_b, fontsize=14)
    ax6.set_ylabel('Image number', fontsize=14)
    ax6.legend(fontsize=12)


    ax1.set_xlim(0, len(lb_data_a) - 1)
    ax2.set_xlim(0, len(lb_data_b) - 1)
    ax1.set_ylim(scatter_ylim_a)
    ax2.set_ylim(scatter_ylim_b)
    ax3.set_xlim(hist_xlim_a)
    ax5.set_xlim(hist_xlim_a)
    ax3.set_ylim(hist_ylim_a)
    ax5.set_ylim(hist_ylim_a)
    ax4.set_xlim(hist_xlim_b)
    ax6.set_xlim(hist_xlim_b)
    ax4.set_ylim(hist_ylim_b)
    ax6.set_ylim(hist_ylim_b)
    

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"The figure has been saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()
        
        
        
def write_percentage_orientation_to_file(file_path,data_dict):
    with open(file_path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        
  
        for header, data_list in data_dict.items():
            writer.writerow([header])
            writer.writerow(data_list)
            

        writer.writerow([]) 
        
     
        t_stat, p_val = stats.ttest_ind(data_dict['bain_area_percentage_list'], data_dict['mar_area_percentage_list'])

        writer.writerow([f"t-statistic for carbide volume fractions: {t_stat}"]) 
        writer.writerow([f"p-value for carbide volume fractions: {p_val}"])     

        if p_val < 0.05:
            writer.writerow(["Conclusion: Reject the null hypothesis. The difference between the group means is statistically significant."]) 
        else:
            writer.writerow(["Conclusion: Fail to reject the null hypothesis. There is no statistically significant difference between the groups."]) 
        
    
        writer.writerow([]) 
            
    
        t_stat, p_val = stats.ttest_ind(data_dict['bainite_k_list'], data_dict['martensite_k_list'])

    
        writer.writerow([f"t-statistic for carbide orientations: {t_stat}"]) 
        writer.writerow([f"p-value for carbide orientations: {p_val}"])    

        if p_val < 0.05:
            writer.writerow(["Conclusion: Reject the null hypothesis. The difference between the group means is statistically significant."]) 
        else:
            writer.writerow(["Conclusion: Fail to reject the null hypothesis. There is no statistically significant difference between the groups."]) 