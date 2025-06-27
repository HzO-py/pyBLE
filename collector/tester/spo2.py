from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
import numpy as np
# **1. 低通滤波（去高频噪声）**
def butter_lowpass_filter(data, cutoff=5, fs=25, order=4):
    """Butterworth 低通滤波"""
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# **2. 高通滤波（去基线漂移）**
def butter_highpass_filter(data, cutoff=0.5, fs=25, order=4):
    """Butterworth 高通滤波"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def butter_bandpass_filter(data, lowcut=0.5, highcut=5, fs=25, order=4):
    """Butterworth 带通滤波"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# def trailing_moving_average(data, window_size=5):
#     data = np.asarray(data, dtype=float)
#     n = len(data)
#     out = np.zeros_like(data)
#     half = window_size // 2

#     for i in range(n):
#         start = max(0, i - half)
#         end   = min(n, i + half + 1) 
#         out[i] = data[start:end].mean()
#     return out

# def load_processed_spo2(fp, overlap=25, multiplier=5, win=5, skip_head=1000,alpha0=1.0):
#     # 1) 读 + 预处理
#     df = read_csv_skip_header_if_needed(fp)
#     df = time_stamp_process(df)   # 需产生 df['local_time'] 的 datetime 列
    
#     ir = df['irCnt'].values[skip_head:-500]
#     red = df['redCnt'].values[skip_head:-500]
#     times = df['local_time'][skip_head:-500]
    
#     # 2) 得到 raw spo2 + times
#     raw, t = process_spo2(ir, red, times,
#                           buffer_size=overlap*multiplier,
#                           overlap_size=overlap,alpha0=alpha0)
#     # 3) 平滑
#     smooth = trailing_moving_average(raw, window_size=win)
#     return t, smooth

# from io import BytesIO
# import itertools
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import pandas as pd

# from matplotlib.widgets import Slider
# from scipy.signal import butter, filtfilt, detrend
# # import mplcursors
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from PIL import Image, ImageDraw, ImageFont

# def high_pass_filter(signal, cutoff=0.5, fs=25, order=3, restore_dc=True):
#     nyq = 0.5 * fs  # 奈奎斯特频率
#     normal_cutoff = cutoff / nyq

#     # 设计 Butterworth 高通滤波器
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)

#     # 记录 DC 分量
#     original_dc = np.mean(signal) if restore_dc else 0

#     # 应用滤波
#     filtered_signal = filtfilt(b, a, signal)

#     # 恢复 DC 分量
#     return filtered_signal + original_dc

# def plot_peaks(ir_buffer, red_buffer, n_npks):
#     fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
#     # IR 信号图
#     axs[0].plot(ir_buffer, label="IR Signal")
#     axs[0].plot(n_npks, ir_buffer[n_npks], "rx", label="Detected Peaks")
#     axs[0].set_title("IR Signal with Detected Peaks")
#     axs[0].set_ylabel("Amplitude")
#     axs[0].legend()
    
#     # Red 信号图
#     axs[1].plot(red_buffer, label="Red Signal")
#     # 如果两个信号同步，可以使用同一组峰值索引
#     axs[1].plot(n_npks, red_buffer[n_npks], "rx", label="Detected Peaks")
#     axs[1].set_title("Red Signal with Detected Peaks")
#     axs[1].set_xlabel("Sample Index")
#     axs[1].set_ylabel("Amplitude")
#     axs[1].legend()
    
#     plt.tight_layout()
#     plt.show()

# # **1. 低通滤波（去高频噪声）**
# def butter_lowpass_filter(data, cutoff=8, fs=25, order=4):
#     """Butterworth 低通滤波"""
#     nyquist = 0.5 * fs  # 奈奎斯特频率
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return filtfilt(b, a, data)

# # **2. 高通滤波（去基线漂移）**
# def butter_highpass_filter(data, cutoff=0.5, fs=25, order=4):
#     """Butterworth 高通滤波"""
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
#     return filtfilt(b, a, data)

# def butter_bandpass_filter(data, lowcut=0.5, highcut=8, fs=25, order=4):
#     """Butterworth 带通滤波"""
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return filtfilt(b, a, data)

# # **3. 平均滤波（去局部噪声）**
# def moving_average_filter(data, window_size=4):
#     """滑动均值滤波"""
#     return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# def preprocess(signal):
#     # signal=butter_highpass_filter(signal)
#     # signal=moving_average_filter(signal)
#     signal=butter_lowpass_filter(signal)
#     # signal=high_pass_filter(signal)
#     return signal

def maxim_ratio_extract(ir_segment, red_segment):
    ir_segment=np.array(ir_segment)
    red_segment=np.array(red_segment)
    peaks, _ = find_peaks(-ir_segment, distance=8, prominence=25)

    ratio_list = []

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]

        # IR 峰值和插值 valley
        ir_max_idx = np.argmax(ir_segment[start:end]) + start
        ir_valley_est = ir_segment[start] + (ir_segment[end] - ir_segment[start]) * ((ir_max_idx - start) / (end - start))
        ir_ac = ir_segment[ir_max_idx] - ir_valley_est
        ir_dc = ir_segment[ir_max_idx]

        # RED 同理
        red_max_idx = np.argmax(red_segment[start:end]) + start
        red_valley_est = red_segment[start] + (red_segment[end] - red_segment[start]) * ((red_max_idx - start) / (end - start))
        red_ac = red_segment[red_max_idx] - red_valley_est
        red_dc = red_segment[red_max_idx]

        if ir_dc > 0 and red_dc > 0:
            ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
            ratio_list.append(ratio)

    if len(ratio_list) == 0:
        return np.nan
    mean_ratio = np.mean(ratio_list)
    return -round(mean_ratio, 3)

def maxim_ratio_extract_rms(ppg_ir, ppg_red):
    ppg_ir=np.array(ppg_ir)
    ppg_red=np.array(ppg_red)
    DC_red = np.mean(ppg_red)
    DC_ir  = np.mean(ppg_ir)
    
    # 2) Compute AC components (RMS around the mean, as one option)
    # AC_red = np.sqrt(np.mean((ppg_red - np.mean(ppg_red))**2))
    # AC_ir  = np.sqrt(np.mean((ppg_ir - np.mean(ppg_ir))**2))
    # AC_red=np.max(ppg_red)-np.min(ppg_red)
    # AC_ir=np.max(ppg_ir)-np.min(ppg_ir)
    AC_red=np.std(ppg_red)
    AC_ir=np.std(ppg_ir)
    
    # 3) Compute ratio R
    #    Avoid division by zero if DC_red or DC_ir is near 0
    if DC_red == 0 or DC_ir == 0:
        return None  # or handle error
    
    R = (AC_red/DC_red ) / (AC_ir/DC_ir )

    return -R

# def maxim_heart_rate_and_oxygen_saturation(ori_ir_buffer, ori_red_buffer,alpha0=1.0):
#     ir_buffer=preprocess(ori_ir_buffer)
#     red_buffer=preprocess(ori_red_buffer)

#     buffer_size = len(ir_buffer)
#     ir_mean = np.mean(ir_buffer)
    
#     # 翻转 IR 信号（使峰值为正）
#     an_x = -1 * (ir_buffer - ir_mean)
    
#     # 使用 scipy 的 find_peaks 来寻找峰值
#     n_npks, _ = find_peaks(an_x, distance=8, prominence=25)

#     # plot_peaks(ir_buffer, red_buffer, n_npks)
    
#     # 将信号转换为 numpy 数组，便于后续操作
#     an_x = np.array(ir_buffer)
#     an_y = np.array(red_buffer)
    
#     ir_ac_list = []
#     ir_dc_list = []
#     r_ac_list = []
#     r_dc_list = []
#     ratio_list = []
    
#     for k in range(len(n_npks) - 1):
#         if n_npks[k + 1] - n_npks[k] > 4:
#             n_x_dc_max = np.max(an_x[n_npks[k]:n_npks[k + 1]])
#             n_y_dc_max = np.max(an_y[n_npks[k]:n_npks[k + 1]])
            
#             n_x_dc_max_idx = np.argmax(an_x[n_npks[k]:n_npks[k + 1]]) + n_npks[k]
#             n_y_dc_max_idx = np.argmax(an_y[n_npks[k]:n_npks[k + 1]]) + n_npks[k]
            
#             # 计算红光信号 AC 值
#             n_y_ac = (an_y[n_npks[k + 1]] - an_y[n_npks[k]]) / (n_npks[k + 1] - n_npks[k])
#             n_y_ac = an_y[n_npks[k]] + n_y_ac * (n_y_dc_max_idx - n_npks[k])
#             n_y_ac = an_y[n_y_dc_max_idx] - n_y_ac
            
#             # 计算 IR 信号 AC 值
#             n_x_ac = (an_x[n_npks[k + 1]] - an_x[n_npks[k]]) / (n_npks[k + 1] - n_npks[k])
#             n_x_ac = an_x[n_npks[k]] + n_x_ac * (n_x_dc_max_idx - n_npks[k])
#             n_x_ac = an_x[n_x_dc_max_idx] - n_x_ac
            
#             if n_x_dc_max > 0 and n_y_dc_max > 0:
#                 ratio = (n_y_ac / n_y_dc_max) / (n_x_ac / n_x_dc_max)
#                 ratio_list.append(ratio)
#                 ir_ac_list.append(n_x_ac)
#                 ir_dc_list.append(n_x_dc_max)
#                 r_ac_list.append(n_y_ac)
#                 r_dc_list.append(n_y_dc_max)
    
#     if len(ratio_list) > 0:
#         # median_idx=np.argsort(ratio_list)[min(int(len(ratio_list) * percentile),len(ratio_list)-1)]
#         # return ir_ac_list[median_idx], ir_dc_list[median_idx], r_ac_list[median_idx], r_dc_list[median_idx], ratio_list[median_idx]
#         return np.mean(ir_ac_list),np.mean(ir_dc_list),np.mean(r_ac_list),np.mean(r_dc_list),np.mean(ratio_list)
#     else:
#         return 0, 0, 0, 0, 0

# def process_spo2(ir_buffer, red_buffer, local_time, buffer_size, overlap_size,alpha0=1.0):
#     """
#     根据给定的窗口长度和重叠长度处理数据，并返回计算得到的 SPO2（ratio）序列
#     """
#     spo2_results = []
#     time_list=[]
#     ir_ac = []
#     ir_dc = []
#     r_ac = []
#     r_dc = []
#     cnt = 0
#     num_buffers = (len(ir_buffer) - buffer_size) // overlap_size
#     ir_buffer=preprocess(ir_buffer)
#     red_buffer=preprocess(red_buffer)

#     for i in range(num_buffers):
#         start_idx = i * overlap_size
#         end_idx = start_idx + buffer_size

#         ir_segment = ir_buffer[start_idx : end_idx]
#         red_segment = red_buffer[start_idx : end_idx]
        
#         # 取分段首个采样对应的 local_time 作为该分段时间
#         segment_time = local_time.iloc[start_idx]
#         ratio_val=maxim_ratio_extract(ir_segment, red_segment)
#         # ir_ac_val, ir_dc_val, r_ac_val, r_dc_val, ratio_val = maxim_heart_rate_and_oxygen_saturation(ir_segment, red_segment,alpha0)
#         # if ir_ac_val == 0:
#         #     cnt += 1
#         #     if len(spo2_results) == 0:
#         #         continue
#         #     ratio_val = spo2_results[-1]
#         #     ir_ac_val = ir_ac[-1]
#         #     ir_dc_val = ir_dc[-1]
#         #     r_ac_val = r_ac[-1]
#         #     r_dc_val = r_dc[-1]
#         spo2_results.append(-ratio_val)  # 注意：此处对 ratio 取了负值
#         # ir_ac.append(ir_ac_val)
#         # ir_dc.append(ir_dc_val)
#         # r_ac.append(r_ac_val)
#         # r_dc.append(r_dc_val)
#         time_list.append(segment_time)
#     return spo2_results,time_list

# def normalize_data(data):
#     """
#     对数据进行 min-max 归一化，映射到 [0,1] 范围
#     """
#     data = np.array(data)
#     if len(data) == 0:
#         return data
#     min_val = np.min(data)
#     max_val = np.max(data)
#     if max_val - min_val == 0:
#         return data
#     return (data - min_val) / (max_val - min_val)

# def trailing_moving_average(data, window_size=5):
#     data = np.asarray(data, dtype=float)
#     n = len(data)
#     out = np.zeros_like(data)
#     half = window_size // 2

#     for i in range(n):
#         start = max(0, i - half)
#         end   = min(n, i + half + 1) 
#         out[i] = data[start:end].mean()
#     return out

# def time_stamp_process(df,calibration_contec=0):
#     # 1) 将 timestamp 列转换为数值，插值、前向填充
#     df['timestamp'] = (
#         pd.to_numeric(df['timestamp'], errors='coerce')
#           .interpolate(method='linear')
#           .ffill()
#     )
#     # 2) 转成 UTC 时间，再转换到加州时区
#     df['utc_time'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
#     df['local_time'] = df['utc_time'].dt.tz_convert('America/Los_Angeles') - pd.Timedelta(seconds=calibration_contec)

#     # 函数在 df 上就地操作，可选择性返回
#     return df

# def get_aligned_signals(sig1, sig2, path):
#     aligned1 = []
#     aligned2 = []
#     for (i, j) in path:
#         aligned1.append(sig1[i])
#         aligned2.append(sig2[j])
#     return np.array(aligned1), np.array(aligned2)

# def read_csv_skip_header_if_needed(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         first_line = f.readline()
#         # strip 去除首尾空白，lower() 转小写便于匹配
#         if "vital values report" in first_line.strip().lower():
#             # 若第一行包含 'Vital Values Report'，则 skiprows=1
#             df = pd.read_csv(file_path, skiprows=1)
#         else:
#             df = pd.read_csv(file_path)
#     return df



# def collect_csv_files_one_level(root_dir):
#     file_paths = []
#     img_paths=[]
#     # 列出 root_dir 下的所有项
#     for subdir in os.listdir(root_dir):
#         full_subdir = os.path.join(root_dir, subdir)
#         if os.path.isdir(full_subdir):
#             # 在 subdir 目录下找 .csv
#             for f in os.listdir(full_subdir):
#                 if f.lower().endswith('.csv'):
#                     file_paths.append(os.path.join(full_subdir, f))
#                     img_paths.append(os.path.join(full_subdir, subdir+'_CONTEC.png'))
#     return file_paths,img_paths
# # --- 主程序 ---

# def main():
#     root_dir = r"../spo2_test_set"
#     file_paths,img_files = collect_csv_files_one_level(root_dir)
#     print("Found CSV files:", file_paths)

#     overlap_size = 25
#     multipliers = [5]  # 每个 multiplier 单独出一个 PDF
#     window_sizes = [10]

#     # 2) 读取并处理所有 CSV
#     dfs = []
#     for fp in file_paths:
#         df = read_csv_skip_header_if_needed(fp)  # 若首行含 "Vital Values Report" 则跳过
#         df = time_stamp_process(df,calibration_contec=17)              # 给 df 加 local_time
#         dfs.append(df)

#     # 3) 提取 IR/Red/time
#     channel_names = ['grn2Cnt', 'irCnt', 'redCnt']
#     combos = list(itertools.permutations(channel_names, 2))

#     for combo in combos:
#         ch1, ch2 = combo
#         if ch1!='irCnt' or ch2!='redCnt':
#             continue
#         print(f"Processing combination: {ch1} & {ch2}")
#         buf1_list = [df[ch1].values for df in dfs]
#         buf2_list = [df[ch2].values for df in dfs]
#         time_lists = [df['local_time'] for df in dfs]
#         for m in multipliers:
#             for win in window_sizes:
#                 buffer_size = overlap_size * m
#                 precomputed_data = []  # 每个元素: (spo2_results, time_list) 对于一个文件
#                 for i in range(len(file_paths)):
#                     spo2_results, time_list = process_spo2(
#                         buf1_list[i][500:],  # 可调整截取数据
#                         buf2_list[i][500:],
#                         time_lists[i][500:],
#                         buffer_size,
#                         overlap_size
#                     )
#                     smoothed_spo2 = trailing_moving_average(spo2_results, window_size=win)
#                     precomputed_data.append((smoothed_spo2, time_list))

#                 # 生成每个文件对应的页面：左侧是绘图（CSV处理图），右侧是 CONTEC.png 图片
#                 pages = []
#                 for i in range(len(file_paths)):
#                     # --- 生成 CSV 图 --- 
#                     fig, ax = plt.subplots(figsize=(6, 4))
#                     spo2_data, time_data = precomputed_data[i]
#                     file_name = os.path.basename(file_paths[i])
#                     folder_name = os.path.basename(os.path.dirname(file_paths[i]))
#                     x_indices = np.arange(len(spo2_data))

#                     a=2.655653
#                     b=16.497660
#                     c=101.727465
#                     spo2_data=a*spo2_data**2+b*spo2_data+c

#                     line, = ax.plot(time_data,spo2_data, linestyle='-', picker=5)
#                     ax.set_title(f"Folder: {folder_name} | File: {file_name}\n")
#                     ax.set_ylabel("-R Value")
#                     # ax.legend()
#                     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
#                     ax.xaxis.set_major_locator(mdates.SecondLocator(interval=60))
#                     ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=20))
#                     fig.autofmt_xdate()

#                     plt.show()

#                     # cursor = mplcursors.cursor(line, hover=True)
#                     # @cursor.connect("add")
#                     # def _(sel):
#                     #     # 找到点击点最近的真实索引（通过距离最小值）
#                     #     x_click = sel.target[0]
#                     #     idx = int(np.argmin(np.abs(x_indices - x_click)))
#                     #     sel.annotation.set_text(
#                     #         f'id: {idx}\n'
#                     #         f'time: {time_data[idx].strftime("%H:%M:%S")}\n'
#                     #         f'SPO₂: {spo2_data[idx]:.1f}%\n'
#                     #     )
#                     #     sel.annotation.get_bbox_patch().set_alpha(0.85)
#                     # plt.tight_layout()
#                     # plt.show()

#                     # 将图保存到内存中的 BytesIO 对象
#                     buf = BytesIO()
#                     fig.savefig(buf, format='png', bbox_inches='tight')
#                     plt.close(fig)
#                     buf.seek(0)
#                     plot_img = Image.open(buf)

#                     # --- 读取对应的 CONTEC.png 图片 ---
#                     # 根据 img_files[i] 取得对应图片，如果不存在则生成一个空白图
#                     contec_path = img_files[i]
#                     if os.path.exists(contec_path):
#                         try:
#                             contec_img = Image.open(contec_path)
#                             if contec_img.mode != "RGB":
#                                 contec_img = contec_img.convert("RGB")
#                         except Exception as e:
#                             print(f"Error reading {contec_path}: {e}")
#                             # 使用空白图代替
#                             contec_img = Image.new("RGB", plot_img.size, (255,255,255))
#                     else:
#                         # 若文件不存在，生成一个空白图
#                         contec_img = Image.new("RGB", plot_img.size, (255,255,255))

#                     # --- 对左右图像进行水平拼接 ---
#                     # 调整两图高度相同（取最大高度）
#                     h1 = plot_img.height
#                     h2 = contec_img.height
#                     target_height = max(h1, h2)
#                     # 若不一致，则将两图缩放到 target_height（保持宽高比）
#                     if h1 != target_height:
#                         ratio = target_height / h1
#                         new_width = int(plot_img.width * ratio)
#                         plot_img = plot_img.resize((new_width, target_height), Image.Resampling.LANCZOS)
#                     if h2 != target_height:
#                         ratio = target_height / h2
#                         new_width = int(contec_img.width * ratio)
#                         contec_img = contec_img.resize((new_width, target_height), Image.Resampling.LANCZOS)
#                     # 拼接图片
#                     total_width = plot_img.width + contec_img.width
#                     combined_page = Image.new("RGB", (total_width, target_height), (255,255,255))
#                     combined_page.paste(plot_img, (0,0))
#                     combined_page.paste(contec_img, (plot_img.width, 0))
#                     pages.append(combined_page)

#                 # 将所有页面保存为一个 PDF
#                 if pages:
#                     min_h = min(p.height for p in pages)

#                     # 2) 等比缩放所有高度超过 min_h 的页面
#                     scaled_pages = []
#                     for p in pages:
#                         if p.height > min_h:
#                             scale = min_h / p.height
#                             new_w = int(p.width * scale)
#                             p = p.resize((new_w, min_h), Image.Resampling.LANCZOS)
#                         # 如果你也想把更矮的页面放大到 min_h，可以把上面 if 换成 if p.height != min_h
#                         scaled_pages.append(p)

#                     # 3) 用 scaled_pages 来保存 PDF
#                     if scaled_pages:
#                         out_name = f"imgs/spo2_{ch1}_{ch2}_m{m}_win{win}_checkout.pdf"
#                         scaled_pages[0].save(
#                             out_name,
#                             save_all=True,
#                             append_images=scaled_pages[1:],
#                             resolution=100.0
#                         )
#                         print(f"Saved scaled PDF: {out_name}")


# if __name__ == "__main__":
#     main()

