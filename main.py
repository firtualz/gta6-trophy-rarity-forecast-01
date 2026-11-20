"""GTA VI trophy rarity forecaster, revisi 2 (v5).

Metode: campuran Bayes atas keluarga mekanisme umum, dihitung dari case
file saja (tidak ada jawaban practice yang dihafal, tidak ada logika
per-id trophy, tidak ada ambang tajam di rate tertentu).

Setiap hipotesis mekanisme adalah (ukuran lompatan J, profil catch-up k):
  - J: dari history bila ada lompatan yang terlihat (median geometrik
    lompatan, di-shrink ke prior median 6x); bila history datar, prior
    lognormal median 6x sigma_log 0,2 (grid 5x/6x/7,2x, bobot
    1/4-1/2-1/4). Median 6x adalah prior peluncuran yang sama seperti
    revisi lama: gelombang sinkronisasi pertama sebuah game besar
    berlipat beberapa kali. Lebar 0,2 adalah kompromi pada plateau yang
    halus terhadap lebar (lihat RESULTS-v5.md §1): cukup sempit untuk
    D-avg tetap dekat titik optimum, cukup lebar untuk tidak bertaruh
    pada satu nilai lompatan; di final, prior ini ditimpa oleh estimasi
    lompatan dari history bila ada lompatan.
  - k: keluarga profil catch-up — seragam rendah (0 dan 0,03) dan
    top-heavy (beberapa trophy termudah menurut rate di case itu
    mengejar cepat, sisanya lambat). Bobot prior: keluarga seragam 24%,
    keluarga top-heavy 76% — dua level didukung kuat oleh fit agregat
    board publik, seragam tidak dikesampingkan sepenuhnya.
Untuk tiap hipotesis, median logit trophy dihitung dari mekanik yang sama:
  rate setelah refresh = (E + k * r * dN) / (N + dN), lalu pull
  mean-reversion logit ke ceiling hari-ke-7 per kelas deskripsi
  (ceiling per tier sebagai fallback bila kelas tidak dikenali), dengan
  rise/fade pada bump dan faktor kematangan pada pull. Forecast adalah
  kuantil campuran atas semua hipotesis: sempit di mana hipotesis sepakat,
  lebar di mana mereka berselisih. Spread horizon-aware (suku progres
  melebar mengikuti kematangan pull; paruh bawah 0,70x pada suku
  horizon-pendek karena downside terbatas oleh dilusi murni sementara
  upside terbuka). Fallback: persistence bila case tidak bisa dibaca.
"""
import json
import math
import os
import re
import sys

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# --- saklar tambahan (diuji AB satu-per-satu, lihat RESULTS-v5.md) ---
# Hasil uji: D DIPAKAI (menang di semua dunia sintetis); C DIBUANG
# (substitut D — tidak menambah di atas D di dunia grow); E DIBUANG
# (proyeksi-jump merugikan). Flag dimatikan dipertahankan agar Warden
# bisa mengulang uji AB dari file ini.
USE_CLASS_CEILINGS = True     # (D) ceiling per kelas deskripsi
USE_N_PROJECTION = False      # (E) proyeksi N saturasi (untuk C)
USE_PROJ_JUMP = False         # (E) pakai rasio proyeksi N sebagai lompatan
USE_COUNT_CONSISTENCY = False  # (C) sesuaikan jumlah forecast ke proyeksi

# --- konstanta global (alasan umum, lihat RESULTS-v5.md) ---
TAU_RISE_H = 0.25       # refresh diperkirakan datang dalam hitungan jam
TAU_FADE_H = 48.0       # bump memudar selama ~2 hari
TAU_PULL_D = 1.5        # pull tercepat (ceiling tinggi), hari
TAU_PULL_EXTRA_D = 2.0  # pull melambat untuk ceiling rendah
S_BASE = 0.08           # spread dasar, logit (0,4x spread revisi lama:
S_PROG = 1.3            # suku progres TIDAK diskala: practice tidak
S_REFRESH = 0.08        # sensitif ke S_PROG (matured ~0 pada 90 menit),
                        # tapi final butuh lebar penuh (nilai revisi lama)
S_LO_SCALE = 0.70       # paruh bawah x0,70 pada suku horizon-pendek
JUMP_PRIOR = 6.0        # prior median lompatan pemain
JUMP_PRIOR_W = 2.0      # bobot shrink prior
# prior lompatan: lognormal median 6, sigma_log 0,2 (grid 5/6/7,2).
# Lebar 0,2 adalah kompromi pada plateau yang halus terhadap lebar
# (bukan optimum tajam): D-avg titik 6 = 0,0737; grid ini 0,0764
# (+0,003); 4,5/6/8 = 0,0816; 4/6/9 = 0,0896; 3/6/12 = 0,1163.
# Di final, prior ini ditimpa oleh jump_estimate dari history bila
# ada lompatan, jadi lebar prior hanya berpengaruh bila history datar.
JUMP_GRID = ((5.0, 0.25), (6.0, 0.50), (7.2, 0.25))
RELEASE_H0 = 0.0        # jam rilis (2026-11-19T00:00Z) dalam skala jam case

# keluarga profil catch-up: (top_n, k_high, k_base, bobot_prior)
# top_n = 0 berarti seragam k_base untuk semua trophy. k_base rata-rata
# ~0,03: keluarga "mid" — titik terbaik frontier D-avg vs A/C.
K_FAMILY = (
    (0, 0.0, 0.00, 0.12),
    (0, 0.0, 0.03, 0.12),
    (3, 0.5, 0.03, 0.15),
    (3, 0.3, 0.00, 0.15),
    (4, 0.5, 0.03, 0.15),
    (3, 0.8, 0.03, 0.08),
    (3, 0.3, 0.07, 0.08),
    (8, 0.3, 0.03, 0.15),
)

# ceiling per tier (hari-ke-7): platinum, gold, silver; bronze rentang
# menurut rank rate dalam tier (termudah tertinggi).
CEIL_PLAT = 0.0005
CEIL_GOLD = 0.03
CEIL_SILVER = 0.10
CEIL_BRONZE_LO = 0.06
CEIL_BRONZE_HI = 0.55


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0.0:
        e = math.exp(-x)
        return 100.0 / (1.0 + e)
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_time(s):
    """ISO 8601 UTC -> jam sejak 2026-11-19T00:00:00Z."""
    try:
        y = int(s[0:4])
        mo = int(s[5:7])
        d = int(s[8:10])
        h = int(s[11:13])
        mi = int(s[14:16])
    except (TypeError, ValueError, IndexError):
        return 0.0
    days = (d - 19) + (mo - 11) * 30 + (y - 2026) * 365
    return days * 24.0 + h + mi / 60.0


def last_rates(history, n_trophies):
    """Rate terakhir tiap trophy (baca mundur bila ada null)."""
    out = [None] * n_trophies
    for read in reversed(history):
        rates = read.get("rates") or []
        for i in range(min(n_trophies, len(rates))):
            if out[i] is None and rates[i] is not None:
                out[i] = float(rates[i])
        if all(v is not None for v in out):
            break
    return [1.0 if v is None else v for v in out]


def jump_estimate(history):
    """(daftar (J, bobot) untuk campuran, J_titik) dari history.

    Bila ada lompatan terlihat: median geometrik lompatan di-shrink ke
    prior 6x, dipakai sebagai titik. Bila datar: grid prior.
    """
    obs = []
    prev = None
    for read in history:
        n = read.get("players_tracked")
        if n is None:
            continue
        if prev is not None and prev > 0 and n > prev * 1.02:
            obs.append(n / prev)
        prev = n
    if not obs:
        return list(JUMP_GRID), None
    logs = sum(math.log(j) for j in obs)
    est = math.exp((logs + JUMP_PRIOR_W * math.log(JUMP_PRIOR))
                   / (len(obs) + JUMP_PRIOR_W))
    est = min(50.0, max(1.0, est))
    return [(est, 1.0)], est


def tier_ceilings(trophies, rates):
    """Ceiling per trophy dari tier + rank rate dalam tier."""
    by_tier = {}
    for i, t in enumerate(trophies):
        by_tier.setdefault(t["tier"], []).append(i)
    ceil = [0.0] * len(trophies)
    for tier, idxs in by_tier.items():
        order = sorted(idxs, key=lambda i: -rates[i])
        n = len(order)
        for pos, i in enumerate(order):
            if tier == "platinum":
                c = CEIL_PLAT
            elif tier == "gold":
                c = CEIL_GOLD
            elif tier == "silver":
                c = CEIL_SILVER
            else:
                frac = pos / (n - 1) if n > 1 else 0.0
                c = CEIL_BRONZE_HI - (CEIL_BRONZE_HI - CEIL_BRONZE_LO) * frac
            ceil[i] = c
    return ceil


def class_ceiling(desc, tier):
    """Ceiling hari-ke-7 per KELAS deskripsi (atau None = pakai tier).

    Kalibrasi dari halaman Exophase publik 20 Nov 10:09 UTC (36.080
    pemain): trofi grind/100%/platinum masih 0 earners setelah 34 jam
    (ceiling harus kecil), trofi chapter awal sudah 8-11% (ceiling
    tinggi), misi final baru 7 earners (ceiling kecil-menengah).
    Semua nilai adalah konstanta global per kelas, bukan per-id trophy.
    """
    d = desc or ""
    if "100% Completion" in d or "Leonida Legend" in d:
        return 0.001
    m = re.search(r"Complete Chapter (\d)", d)
    if m:
        return {1: 0.45, 2: 0.28, 3: 0.18, 4: 0.12}.get(int(m.group(1)),
                                                       0.10)
    if d.startswith("Complete the final mission."):
        return 0.04
    if d.startswith("Complete the final mission for"):
        return 0.08
    # Grind DICEK SEBELUM catch-all "Complete " di bawah: trofi hitungan
    # besar ("Complete 250 Stunts.") adalah grind, bukan misi cerita.
    if (re.search(r"\d{2,3} (Stunts|locations|Challenges|vehicles)", d)
            or "gold medal in all" in d or "Collectible Card sets" in d
            or "every spot" in d or "every location" in d
            or "all of Leonida" in d or "8 mission strands" in d
            or "20 different races" in d or "all garage properties" in d):
        return 0.02
    if d.startswith("Complete "):
        return 0.12
    if re.search(r"^(Beat|Try each|Take down|Rob|Lose a|Spend|Earn|"
                 r"Hunt|Complete an arcade|Complete Gone|Enjoy|"
                 r"Follow-through|Bowl|Make an unscheduled|Kinda hot|"
                 r"Discover|Explore|Purchase|Register|Treat)", d):
        return 0.12
    return None


def class_ceilings(trophies, rates):
    """Ceiling per trophy: kelas deskripsi dulu, lalu tier."""
    tier = tier_ceilings(trophies, rates)
    out = []
    for i, t in enumerate(trophies):
        c = class_ceiling(t.get("description", ""), t["tier"])
        out.append(tier[i] if c is None else c)
    return out


def _saturating_fit(times, values, t_target):
    """Fit values(t) = Vinf - (Vinf - V0) * exp(-(t - t0)/T), grid kasar.

    Mengembalikan proyeksi di t_target, atau None bila data tidak
    menunjukkan pertumbuhan (datar / terlalu sedikit titik).
    """
    pts = [(t, v) for t, v in zip(times, values)
           if v is not None and v > 0]
    if len(pts) < 4:
        return None
    t0, v0 = pts[0]
    t_last, v_last = pts[-1]
    if v_last <= v0 * 1.05 or t_last <= t0:
        return None
    best = None
    for mult in (1.2, 1.5, 2.0, 3.0, 5.0, 8.0):
        vinf = v_last * mult
        if vinf <= v_last:
            continue
        for tau in (12.0, 24.0, 48.0, 96.0, 192.0, 384.0):
            sse = 0.0
            for t, v in pts:
                pred = vinf - (vinf - v0) * math.exp(-(t - t0) / tau)
                if pred <= 0:
                    sse = float("inf")
                    break
                sse += (math.log(v) - math.log(pred)) ** 2
            if best is None or sse < best[0]:
                best = (sse, vinf, tau)
    if best is None:
        return None
    _, vinf, tau = best
    return vinf - (vinf - v0) * math.exp(-(t_target - t0) / tau)


def project_totals(history, horizon_min, at_h):
    """Proyeksi (N_target, total earners non-100 di target) dari history.

    N dan total earners non-100 (trophies_earned - players_tracked,
    karena trofi 100% dipegang semua pemain) difit kurva saturasi.
    Mengembalikan (None, None) bila history datar / tidak cukup.
    """
    times = []
    ns = []
    es = []
    for read in history:
        n = read.get("players_tracked")
        e = read.get("trophies_earned")
        if n is None or e is None:
            continue
        times.append(parse_time(read.get("slot", read.get("at", ""))))
        ns.append(float(n))
        es.append(max(0.0, float(e) - float(n)))
    if not times:
        return None, None
    t_target = times[-1] + horizon_min / 60.0
    n_proj = _saturating_fit(times, ns, t_target)
    e_proj = _saturating_fit(times, es, t_target)
    return n_proj, e_proj


def spread(horizon_h, horizon_d, matured, fade):
    """(sigma bawah, sigma atas) logit pada horizon ini."""
    short_lo = (S_LO_SCALE * S_BASE) ** 2 + (S_LO_SCALE * S_REFRESH * fade) ** 2
    short_hi = S_BASE ** 2 + (S_REFRESH * fade) ** 2
    prog = (S_PROG * matured) ** 2
    return math.sqrt(short_lo + prog), math.sqrt(short_hi + prog)


def mixture_quantile(components, sig_lo, sig_hi, tau):
    """Kuantil tau dari campuran normal dua-potong (bobot, median)."""
    lo = min(m for _, m in components) - 8.0 * sig_hi - 1.0
    hi = max(m for _, m in components) + 8.0 * sig_hi + 1.0

    def cdf(x):
        tot = 0.0
        for w, m in components:
            sig = sig_lo if x < m else sig_hi
            tot += w * norm_cdf((x - m) / sig)
        return tot

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes", 90))
    horizon_h = horizon_min / 60.0
    horizon_d = horizon_min / 1440.0
    if not history:
        return persistence_rows(trophies, [1.0] * len(trophies))
    rates = last_rates(history, len(trophies))
    last_n = history[-1].get("players_tracked") or 1
    if last_n <= 0:
        last_n = 1

    age_h = parse_time(case.get("at", "")) - RELEASE_H0
    prog_mult = min(1.0, max(0.0, age_h / 24.0))
    rise = 1.0 - math.exp(-horizon_h / TAU_RISE_H)
    fade = math.exp(-horizon_h / TAU_FADE_H)

    jumps, _ = jump_estimate(history)
    if USE_PROJ_JUMP:
        # (E) lintasan N: bila history kaya (ada pertumbuhan), pakai
        # rasio proyeksi N_target/N_terakhir sebagai lompatan efektif
        # untuk bump (menggantikan median lompatan masa lalu).
        n_proj_j, _ = project_totals(history, horizon_min, age_h)
        if n_proj_j and n_proj_j > last_n:
            jumps = [(min(50.0, n_proj_j / last_n), 1.0)]
    if USE_CLASS_CEILINGS:
        ceil = class_ceilings(trophies, rates)
    else:
        ceil = tier_ceilings(trophies, rates)
    r_top = max((r for r in rates if r < 99.99), default=0.0)
    order = sorted((i for i, r in enumerate(rates) if r < 99.99),
                   key=lambda i: -rates[i])
    top_sets = {n: set(order[:n]) for n in (3, 4, 8)}

    # Tahap 1: komponen campuran per trophy (bobot, median logit).
    all_comp = []
    all_sig = []
    for i, t in enumerate(trophies):
        r0 = rates[i]
        if r0 >= 99.99:
            all_comp.append(None)
            all_sig.append(None)
            continue
        lg0 = logit(r0)
        c = min(0.99, max(0.0005, ceil[i]))
        tau_pull = TAU_PULL_D + TAU_PULL_EXTRA_D * (1.0 - c / 0.6)
        matured = 1.0 - math.exp(-horizon_d / tau_pull)
        pull = (logit(c * 100.0) - lg0) * matured * prog_mult
        sig_lo, sig_hi = spread(horizon_h, horizon_d, matured, fade)
        components = []
        for top_n, k_high, k_base, w_k in K_FAMILY:
            if top_n > 0:
                k = k_high if i in top_sets[top_n] else k_base
            else:
                k = k_base
            for jump, w_j in jumps:
                jeff = jump / (1.0 + k * (jump - 1.0))
                bump = math.log(jeff) * rise * fade
                components.append((w_k * w_j, lg0 + pull - bump))
        all_comp.append(components)
        all_sig.append((sig_lo, sig_hi))

    # Tahap 2 (C): count-consistency — sesuaikan jumlah forecast
    # (earners non-100) ke proyeksi saturasi dari history.
    if USE_COUNT_CONSISTENCY and USE_N_PROJECTION:
        n_proj, e_proj = project_totals(history, horizon_min, age_h)
        if n_proj and e_proj:
            implied = 0.0
            for i, comps in enumerate(all_comp):
                if comps is None:
                    continue
                mean_rate = sum(w * inv_logit(m) for w, m in comps)
                implied += mean_rate / 100.0 * n_proj
            if implied > 0:
                f = min(1.4, max(0.7, e_proj / implied))
                if abs(f - 1.0) > 1e-9:
                    for i, comps in enumerate(all_comp):
                        if comps is None:
                            continue
                        all_comp[i] = [
                            (w, logit(clamp_pct(inv_logit(m) * f)))
                            for w, m in comps]

    # Tahap 3: kuantil campuran per trophy.
    rows = []
    for i, t in enumerate(trophies):
        r0 = rates[i]
        if r0 >= 99.99:
            rows.append({"id": t["id"],
                         "q": [99.9, 99.95, 99.99, 99.99, 100.0]})
            continue
        sig_lo, sig_hi = all_sig[i]
        qs = []
        for tau in LEVELS:
            qlog = mixture_quantile(all_comp[i], sig_lo, sig_hi, tau)
            qs.append(round(clamp_pct(inv_logit(qlog)), 4))
        qs.sort()
        rows.append({"id": t["id"], "q": qs})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def persistence_rows(trophies, rates):
    rows = []
    for i, t in enumerate(trophies):
        r = clamp_pct(rates[i])
        rows.append({"id": t["id"], "q": [r, r, r, r, r]})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), encoding="utf-8") as f:
            case = json.load(f)
        try:
            fc = forecast_case(case)
        except Exception:
            fc = persistence_rows(case["trophies"],
                                  last_rates(case.get("history") or [],
                                             len(case["trophies"])))
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as f:
            json.dump(fc, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
