import sys
import ast
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Standard font settings for English
plt.rcParams['font.family'] = 'sans-serif'

def load_alog(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            data = ast.literal_eval(content)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def get_nested_value(data, keys):
    for key in keys:
        if key in data: return data[key]
    sub_dicts = [data.get('temps', {}), data.get('extra', {}), data.get('computed', {})]
    for sub in sub_dicts:
        for key in keys:
            if key in sub: return sub[key]
    return None

def force_extract_series_at_index(data, key_candidates, index=0):
    raw_data = get_nested_value(data, key_candidates)
    if not raw_data: return None
    arr = np.array(raw_data)
    if arr.ndim == 1: return arr
    if arr.ndim == 2:
        if index < arr.shape[0]: return arr[index]
        return arr[0]
    return None

def pad_start_to_zero(times, data_series):
    """ Pad start to t=0 to visualize initial value """
    if data_series is None or len(data_series) == 0:
        return times, data_series
    times = np.array(times)
    data_series = np.array(data_series)
    if len(times) > 0 and times[0] > 0.0:
        times = np.insert(times, 0, 0.0)
        data_series = np.insert(data_series, 0, data_series[0])
    return times, data_series

def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"

def calculate_smooth_ror(temps, times, span_sec=7):
    temps = np.array(temps)
    times = np.array(times)
    dt_avg = np.mean(np.diff(times))
    if dt_avg == 0: return np.zeros_like(temps)
    window_size = int(span_sec / dt_avg)
    if window_size < 1: window_size = 1
    gradient = np.gradient(temps, times)
    ror_raw = gradient * 60
    if window_size > 1:
        window = np.ones(window_size) / window_size
        ror_smooth = np.convolve(ror_raw, window, mode='same')
        ror_smooth[:window_size] = ror_raw[:window_size]
        ror_smooth[-window_size:] = ror_raw[-window_size:]
        return ror_smooth
    else:
        return ror_raw

def get_stabilized_start_value(values):
    """ Get a stabilized initial value (skipping transient noise) """
    if values is None or len(values) == 0:
        return 0, 0
    target_idx = min(len(values) - 1, 15)
    val = values[target_idx]
    return val, target_idx

def render_roast_graph(alog_path, output_path):
    data = load_alog(alog_path)
    
    # 1. Main Data
    bt = force_extract_series_at_index(data, ['temp1', 'BT'])
    et = force_extract_series_at_index(data, ['temp2', 'ET'])
    if bt is None:
        print("Error: No BT data found.")
        sys.exit(1)

    timex = get_nested_value(data, ['timex'])
    if timex:
        time_sec = np.array(timex)
    else:
        interval = data.get('samplinginterval', 1.0)
        time_sec = np.arange(len(bt)) * interval
    
    limit = min(len(time_sec), len(bt))
    time_sec = time_sec[:limit]
    bt = bt[:limit]
    if et is not None: et = et[:limit]

    # 2. Control Data
    power_data = force_extract_series_at_index(data, ['extratemp1', 'extra1'], index=0)
    fan_data   = force_extract_series_at_index(data, ['extratemp2', 'extra2'], index=0)

    if power_data is not None: power_data = power_data[:limit]
    if fan_data is not None: fan_data = fan_data[:limit]

    # Padding
    p_times, p_vals = time_sec, power_data
    if power_data is not None:
        p_times, p_vals = pad_start_to_zero(time_sec, power_data)

    f_times, f_vals = time_sec, fan_data
    if fan_data is not None:
        f_times, f_vals = pad_start_to_zero(time_sec, fan_data)

    # 3. RoR Calculation
    target_temp = et if et is not None and len(et) > 0 else bt
    ror_label = 'ΔET (7s)' if et is not None and len(et) > 0 else 'ΔBT (7s)'
    ror = calculate_smooth_ror(target_temp, time_sec, span_sec=7)

    # 4. Events
    computed = data.get('computed', {})
    tp_time = computed.get('TP_time', 0)
    fcs_time = computed.get('FCs_time', 0)
    drop_time = computed.get('DROP_time', time_sec[-1])
    tp_temp = computed.get('TP_BT', 0)
    fcs_temp = computed.get('FCs_BT', 0)
    drop_temp = computed.get('DROP_BT', bt[-1])
    charge_temp = computed.get('CHARGE_BT', bt[0])

    over_150 = np.where(bt >= 150)[0]
    dry_time = time_sec[over_150[0]] if len(over_150) > 0 else (tp_time + (fcs_time-tp_time)/2 if fcs_time > 0 else 0)

    # 5. Drawing
    fig, ax1 = plt.subplots(figsize=(14, 9))
    ax2 = ax1.twinx()

    ax1.set_facecolor('white')
    ax1.grid(True, which='major', color='#e0e0e0', linestyle='-')

    if fcs_time > 0:
        ax1.axvspan(fcs_time/60, drop_time/60, color='#E6CEF2', alpha=0.4)

    # --- Plot: Control Logs ---
    if p_vals is not None:
        ax1.plot(p_times/60.0, p_vals, color='#B22222', linewidth=2.0, drawstyle='steps-post', alpha=0.8, label='Burner')
        if len(p_vals) > 0:
            stabilized_val, idx = get_stabilized_start_value(p_vals)
            display_val = int(stabilized_val / 10) if (stabilized_val >= 10 and stabilized_val % 10 == 0) else int(stabilized_val)
            arrow_x = p_times[idx] / 60.0
            ax1.annotate(f"Start: {display_val}", xy=(arrow_x, stabilized_val), xytext=(15, 10), textcoords='offset points',
                         fontsize=9, color='#B22222', fontweight='bold', arrowprops=dict(arrowstyle='->', color='#B22222'), clip_on=False)

    if f_vals is not None:
        ax1.plot(f_times/60.0, f_vals, color='#4682B4', linewidth=2.0, drawstyle='steps-post', alpha=0.8, label='Air')
        if len(f_vals) > 0:
            stabilized_val, idx = get_stabilized_start_value(f_vals)
            display_val = int(stabilized_val / 10) if (stabilized_val >= 10 and stabilized_val % 10 == 0) else int(stabilized_val)
            arrow_x = f_times[idx] / 60.0
            ax1.annotate(f"Start: {display_val}", xy=(arrow_x, stabilized_val), xytext=(15, -15), textcoords='offset points',
                         fontsize=9, color='#4682B4', fontweight='bold', arrowprops=dict(arrowstyle='->', color='#4682B4'), clip_on=False)

    # --- Main Plot ---
    ax2.plot(time_sec/60.0, ror, color='#9370DB', linewidth=1.2, alpha=0.9, label=ror_label, zorder=10)
    if et is not None:
        ax1.plot(time_sec/60.0, et, color='#C71585', linewidth=2.5, label='ET', zorder=11)
    ax1.plot(time_sec/60.0, bt, color='#008B8B', linewidth=3.0, label='BT', zorder=12)

    # --- Annotations ---
    annot_opts = dict(xycoords='data', textcoords='offset points', fontsize=9, arrowprops=dict(arrowstyle='-', color='black'))
    
    ax1.annotate('CHARGE', xy=(0, charge_temp), xytext=(10, 20), **annot_opts)
    if tp_time > 0:
        ax1.annotate(f"TP {format_time(tp_time)}", xy=(tp_time/60, tp_temp), xytext=(20, -10), **annot_opts)
    if fcs_time > 0:
        ax1.annotate(f"FCs {format_time(fcs_time)}", xy=(fcs_time/60, fcs_temp), xytext=(-20, -30), **annot_opts)
    
    # DROP TIME (with bbox)
    ax1.annotate(f"DROP {format_time(drop_time)}", 
                 xy=(drop_time/60, drop_temp), 
                 xytext=(-45, 5), 
                 textcoords='offset points',
                 fontsize=9,
                 arrowprops=dict(arrowstyle='-', color='black'),
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    
    # END TEMPERATURE (with bbox)
    ax1.annotate(f"{drop_temp:.1f}C", 
                 xy=(drop_time/60, drop_temp), 
                 xytext=(0, 20), 
                 textcoords='offset points', 
                 ha='center', 
                 fontsize=12, 
                 fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # --- Phase Bars ---
    if dry_time > 0 and fcs_time > 0:
        y_bar = 265
        h_bar = 8
        pct_dry = (dry_time / drop_time) * 100
        ax1.barh(y_bar, dry_time/60, height=h_bar, left=0, color='#cccccc', alpha=0.5, align='center')
        ax1.text(dry_time/120, y_bar, f"{format_time(dry_time)}\n{pct_dry:.1f}%", ha='center', va='center', fontsize=8)
        
        dur_mid = fcs_time - dry_time
        pct_mid = (dur_mid / drop_time) * 100
        ax1.barh(y_bar, dur_mid/60, height=h_bar, left=dry_time/60, color='#e6e6e6', alpha=0.8, align='center')
        ax1.text((dry_time + dur_mid/2)/60, y_bar, f"{format_time(dur_mid)}\n{pct_mid:.1f}%", ha='center', va='center', fontsize=8)
        
        dur_fin = drop_time - fcs_time
        pct_fin = (dur_fin / drop_time) * 100
        ax1.barh(y_bar, dur_fin/60, height=h_bar, left=fcs_time/60, color='#E6CEF2', alpha=1.0, align='center')
        ax1.text((fcs_time + dur_fin/2)/60, y_bar, f"{format_time(dur_fin)}\n{pct_fin:.1f}%", ha='center', va='center', fontsize=8)

    # --- Axes & Legends & Title ---
    ax1.set_ylim(0, 280)
    ax1.set_xlim(0, (drop_time/60)+0.5)
    ax1.set_ylabel('Temperature (C) / Power (%)', fontsize=12)
    
    # TOTAL TIME at Bottom
    ax1.set_xlabel(f'Time (min)   [ Total Time: {format_time(drop_time)} ]', fontsize=11, fontweight='bold', color='#333333')
    
    ax2.set_ylim(-5, 25)
    ax2.set_ylabel(f'RoR ({ror_label}) (C/min)', color='#9370DB', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#9370DB')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(1, 0.5), framealpha=0.8)

    # TITLE: Use Filename
    filename = os.path.basename(alog_path)
    plt.title(f"{filename}", fontsize=16, loc='left', color='purple', pad=15)
    
    plt.figtext(0.9, 0.9, "generated by arigato-coffee actions", ha='right', fontsize=8, color='grey')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    print(f"Graph generated: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python render_graph.py <input.alog> <output.png>")
        sys.exit(1)
    render_roast_graph(sys.argv[1], sys.argv[2])
