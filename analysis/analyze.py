import os
import glob
import csv
import sys
import math
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================================
# ЗАВАНТАЖЕННЯ
# ============================================================================
def find_latest_file(pattern):
    files = glob.glob(os.path.join("logs", pattern))
    return max(files, key=os.path.getmtime) if files else None


def load_commands(filename, start_time, end_time):
    cmds = []
    if not filename or not os.path.exists(filename):
        return cmds
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                parts = line.strip().split(" - ")
                if len(parts) == 2:
                    ts, c = parts
                    dt = datetime.strptime(ts.replace(",", "."), "%Y-%m-%d %H:%M:%S.%f")
                    if start_time <= dt <= end_time:
                        cmds.append((dt, c))
            except Exception:
                continue
    return cmds


def load_telemetry(filename):
    t, w, tgt, pwm, curr = [], [], [], [], []
    t0 = None
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["Timestamp"])
                if t0 is None:
                    t0 = ts
                t.append((ts - t0).total_seconds())
                w.append(float(row["Omega"]))
                tgt.append(float(row["Target"]))
                pwm.append(float(row["PWM"]))
                curr.append(float(row["Current_A"]))
            except Exception:
                continue
    return t, w, tgt, pwm, curr, t0

# ============================================================================
# СЕГМЕНТАЦІЯ
# ============================================================================
def find_segments(t, tgt, min_duration=0.3, min_samples=8):
    segs, i, n = [], 0, len(tgt)
    while i < n:
        if tgt[i] == 0:
            i += 1
            continue
        val, start = tgt[i], i
        while i < n and tgt[i] == val:
            i += 1
        end = i
        if (end - start) >= min_samples and (t[end - 1] - t[start]) >= min_duration:
            segs.append({'start': start, 'end': end, 'target': val,
                         't_start': t[start], 't_end': t[end - 1], 'freq': None})
    return segs


def attach_frequencies(segs, cmds, t0):
    cf = []
    for ct, c in cmds:
        if c.startswith("cfreq"):
            try:
                cf.append(((ct - t0).total_seconds(), float(c.split()[1])))
            except Exception:
                pass
    cf.sort()
    for s in segs:
        f = None
        for f_t, f_hz in cf:
            if f_t <= s['t_start'] + 0.05:
                f = f_hz
            else:
                break
        s['freq'] = f
    return segs

# ============================================================================
# МЕТРИКИ
# ============================================================================
def find_settling(t, omega, ref, band):
    """Час останнього виходу з ±band навколо ref. None — не встановилось."""
    last_out = -1
    for i in range(len(omega) - 1, -1, -1):
        if abs(omega[i] - ref) > band:
            last_out = i
            break
    if last_out == -1:
        return 0.0
    if last_out == len(omega) - 1:
        return None
    return t[last_out + 1] - t[0]


def count_crossings(omega, target, band):
    """Кількість виходів ω з ±band навколо target (метрика коливань)."""
    inside = False
    n = 0
    for v in omega:
        now_in = abs(v - target) <= band
        if inside and not now_in:
            n += 1
        inside = now_in
    return n


def calculate_metrics(t, omega, target, ignore_init=3):
    n = len(t)
    if n < 8:
        return None

    # Усталене значення — медіана останніх 20%
    last_n = max(8, int(n * 0.2))
    tail = sorted(omega[-last_n:])
    steady_val = tail[len(tail) // 2]
    steady_err = abs(steady_val - target)

    # Ігноруємо перші кілька семплів при аналізі перерегулювання
    seg = omega[ignore_init:] if n > ignore_init else omega

    # Напрямок перехідного процесу (відносно цілі)
    init_val = omega[ignore_init] if n > ignore_init else omega[0]
    if target > init_val + 0.5:
        direction = "up"
    elif target < init_val - 0.5:
        direction = "down"
    else:
        direction = "flat"

    # Перерегулювання — лише в напрямку руху
    if target > 0:
        if direction == "up":
            peak = max(seg)
            overshoot = max(0.0, (peak - target) / target * 100.0)
        elif direction == "down":
            valley = min(seg)
            peak = valley
            overshoot = max(0.0, (target - valley) / target * 100.0)
        else:
            peak = max(seg)
            overshoot = 0.0
    else:
        peak = max(seg)
        overshoot = 0.0

    # Час першого досягнення цілі (±5%)
    band = max(abs(target) * 0.05, 0.1)
    t_reach = None
    for i in range(n):
        if abs(omega[i] - target) <= band:
            t_reach = t[i] - t[0]
            break

    # Час встановлення: до цілі та до фактичного усталеного значення
    settle_tgt = find_settling(t, omega, target, band)
    settle_ss = find_settling(t, omega, steady_val, max(abs(steady_val) * 0.05, 0.1))

    # Кількість перетинів — метрика коливань
    crossings = count_crossings(omega, target, band)
    seg_duration = t[-1] - t[0]
    cross_rate = crossings / seg_duration if seg_duration > 0 else 0

    # RMS і max (на «сталій» частині, без початкового транзієнту)
    rms = math.sqrt(sum((target - w) ** 2 for w in seg) / len(seg))
    max_err = max(abs(target - w) for w in seg)

    return {
        'steady_val': steady_val,
        'overshoot': overshoot,
        'peak': peak,
        'init_val': init_val,
        'direction': direction,
        't_reach': t_reach,
        'reached': t_reach is not None,
        'settle_tgt': settle_tgt,
        'settle_ss': settle_ss,
        'steady_err': steady_err,
        'rms': rms,
        'max_err': max_err,
        'crossings': crossings,
        'cross_rate': cross_rate,
    }

# ============================================================================
# ВИВІД
# ============================================================================
def fmt_t(x):
    return f"{x:.2f}" if x is not None else "—"


def quality_flag(m):
    """Повертає '!' якщо сегмент поганої якості."""
    if m is None:
        return ''
    flags = []
    if m['overshoot'] > 30:
        flags.append('OV')
    if m['settle_ss'] is None or m['settle_ss'] > 3.0:
        flags.append('NS')  # не стабілізувався / повільно
    if m['cross_rate'] > 3.0:
        flags.append('OSC')  # коливання
    return '[' + ','.join(flags) + ']' if flags else ''


def print_per_segment(segs, ms):
    print("\n=== МЕТРИКИ ПО КОЖНОМУ СЕГМЕНТУ ===\n")
    print(f"{'#':>2} {'Freq':>6} {'Target':>6} {'T[с]':>13} {'dir':>5} "
          f"{'Oversh':>7} {'tReach':>7} {'SettlTgt':>9} {'SettlSS':>8} "
          f"{'Steady':>7} {'Cross':>6} {'Rate':>5} {'flag':>9}")
    print("-" * 108)
    for i, (s, m) in enumerate(zip(segs, ms), 1):
        if m is None:
            continue
        freq_s = f"{s['freq']:.0f}Hz" if s['freq'] is not None else "?"
        t_range = f"{s['t_start']:>5.2f}–{s['t_end']:<5.2f}"
        print(f"{i:>2} {freq_s:>6} {s['target']:>6.1f} {t_range:>13} "
              f"{m['direction']:>5} "
              f"{m['overshoot']:>6.1f}% {fmt_t(m['t_reach']):>7} "
              f"{fmt_t(m['settle_tgt']):>9} {fmt_t(m['settle_ss']):>8} "
              f"{m['steady_err']:>7.3f} {m['crossings']:>6} "
              f"{m['cross_rate']:>5.1f} {quality_flag(m):>9}")
    print("\nflag: OV=перерегулювання>30%, NS=не стабілізувався за 3с, OSC=коливання>3 перетини/с")


def print_by_freq(segs, ms):
    groups = {}
    for s, m in zip(segs, ms):
        if m is None or s['freq'] is None:
            continue
        groups.setdefault(s['freq'], []).append(m)
    if not groups:
        return
    print("\n=== УСЕРЕДНЕНІ МЕТРИКИ ПО ЧАСТОТАХ ===\n")
    for freq in sorted(groups):
        g = groups[freq]
        n = len(g)
        reached = sum(1 for m in g if m['reached'])
        stable = sum(1 for m in g if m['settle_ss'] is not None)
        avg_over = sum(m['overshoot'] for m in g) / n
        avg_steady = sum(m['steady_err'] for m in g) / n
        avg_rms = sum(m['rms'] for m in g) / n
        avg_cross = sum(m['cross_rate'] for m in g) / n
        st_ss = [m['settle_ss'] for m in g if m['settle_ss'] is not None]
        st_ss_s = f"{sum(st_ss)/len(st_ss):.2f} с" if st_ss else "—"
        print(f"  {freq:>4.0f} Гц  (сегментів: {n})")
        print(f"     Ціль досягнуто:      {reached}/{n}")
        print(f"     Стабілізувався:      {stable}/{n}   (сер. час {st_ss_s})")
        print(f"     Перерегулювання:     {avg_over:>6.1f} %")
        print(f"     Статична похибка:    {avg_steady:>6.3f} рад/с")
        print(f"     RMS похибка:         {avg_rms:>6.3f} рад/с")
        print(f"     Перетини цілі:       {avg_cross:>6.1f} / с\n")


def print_by_target(segs, ms):
    groups = {}
    for s, m in zip(segs, ms):
        if m is None or s['freq'] is None:
            continue
        groups.setdefault(s['target'], {})[s['freq']] = m
    if not groups:
        return
    print("=== ПОРІВНЯННЯ ПО ЦІЛЯХ (крос-частотне) ===\n")
    for tgt in sorted(groups):
        print(f"  Target = {tgt:.1f} рад/с:")
        for freq in sorted(groups[tgt]):
            m = groups[tgt][freq]
            print(f"    {freq:>4.0f} Гц:  Settl_ss={fmt_t(m['settle_ss']):>5}  "
                  f"Steady={m['steady_err']:.3f}  Oversh={m['overshoot']:>5.1f}%  "
                  f"Crossing_rate={m['cross_rate']:>4.1f}/с")
        print()

# ============================================================================
# ГРАФІК
# ============================================================================
def draw_plot(t, w, tgt, pwm, curr, segs, cmds, t0, max_t, save=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Універсальний аналіз PBC-експерименту', fontsize=14, fontweight='bold')

    ax1.plot(t, w, 'b-', linewidth=1.3, label='ω')
    ax1.plot(t, tgt, 'orange', '--', linewidth=1.3, label='Target')

    colors = {10: '#fff3b0', 100: '#c8e6c9', 500: '#bbdefb', None: '#e0e0e0'}
    seen = set()
    for s in segs:
        color = colors.get(s['freq'], '#e0e0e0')
        label = f"{s['freq']:.0f} Гц" if s['freq'] is not None else None
        kw = {'color': color, 'alpha': 0.35}
        if label and label not in seen:
            kw['label'] = label
            seen.add(label)
        ax1.axvspan(s['t_start'], s['t_end'], **kw)

    vals = list(w) + list(tgt)
    span = max(2.0, 0.15 * (max(vals) - min(vals)))
    ax1.set_ylim(min(min(vals), 0) - span, max(vals) + span)
    ax1.set_ylabel('Швидкість (рад/с)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)

    ax2.plot(t, pwm, 'g-', linewidth=1.2, label='PWM')
    ax2.set_ylabel('PWM (0–255)')
    ax2.set_ylim(-10, 265)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.5)

    ax3.plot(t, curr, 'r-', linewidth=1.2, label='Струм')
    ax3.set_ylabel('Струм (А)')
    ax3.set_xlabel('Час (с)')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.5)

    for ct, ctext in cmds:
        rel = (ct - t0).total_seconds()
        if 0 <= rel <= max_t:
            for ax in (ax1, ax2, ax3):
                ax.axvline(rel, color='magenta', linestyle='--', alpha=0.35, linewidth=0.8)
            ax1.text(rel + 0.15, ax1.get_ylim()[1] * 0.85, ctext,
                     color='magenta', rotation=90, va='top', fontsize=7.5,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax3.set_xlim(0, max_t + 1)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
        print(f"\n[OK] Графік збережено: {save}")
    else:
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n[ІНФО] Аналіз завершено.")

# ============================================================================
# MAIN
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Універсальний аналіз PBC-експериментів")
    p.add_argument("file", nargs="?", default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--min-duration", type=float, default=0.3)
    args = p.parse_args()

    print("=== УНІВЕРСАЛЬНИЙ АНАЛІЗ PBC-ЕКСПЕРИМЕНТУ ===\n")

    telemetry = args.file or find_latest_file("*telemetry_*.csv")
    if not telemetry or not os.path.exists(telemetry):
        print("[ПОМИЛКА] Файл телеметрії не знайдено.")
        sys.exit(1)
    commands_f = find_latest_file("commands.log")

    print(f"[OK] Телеметрія: {telemetry}")
    if commands_f:
        print(f"[OK] Команди:    {commands_f}")

    t, w, tgt, pwm, curr, t0 = load_telemetry(telemetry)
    if not t:
        print("[ПОМИЛКА] Файл порожній.")
        sys.exit(1)

    print(f"[OK] Записів: {len(t)}, тривалість: {t[-1]:.1f} с\n")

    cmds = load_commands(commands_f, t0, t0 + timedelta(seconds=t[-1] + 5))
    segs = find_segments(t, tgt, min_duration=args.min_duration)
    if not segs:
        print("[УВАГА] Не знайдено жодного сталого сегмента.")
        sys.exit(0)

    segs = attach_frequencies(segs, cmds, t0)
    ms = [calculate_metrics(t[s['start']:s['end']],
                            w[s['start']:s['end']],
                            s['target']) for s in segs]

    print_per_segment(segs, ms)
    print_by_freq(segs, ms)
    print_by_target(segs, ms)

    if not args.no_plot:
        draw_plot(t, w, tgt, pwm, curr, segs, cmds, t0, t[-1], args.save)


if __name__ == "__main__":
    main()