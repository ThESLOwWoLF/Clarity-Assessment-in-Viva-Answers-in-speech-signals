"""Excel label files + wav matching -> one table of (path, label, recording_id)."""
import glob
import logging
import os
import re

import pandas as pd
import yaml

log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["low", "medium", "high"]  # ordinal order: index = 0, 1, 2


def load_config(path=os.path.join(ROOT, "config.yaml")):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ("excel_dir", "wav_dir", "cache", "model_path", "results_dir"):
        cfg[k] = os.path.join(ROOT, cfg[k])  # no-op if already absolute
    return cfg


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _pick(cols, keys):
    return next((k for k in keys if k in cols), None)


def normalize_label(x):
    """'High'/'h'/0.8 -> 'high'; numeric scores binned at 0.33 / 0.66. None if unparseable."""
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    try:
        v = float(s)
        return "high" if v >= 0.66 else "medium" if v >= 0.33 else "low"
    except ValueError:
        pass
    if "high" in s or s == "h":
        return "high"
    if "med" in s or s == "m":
        return "medium"
    if "low" in s or s == "l":
        return "low"
    return None


def to_wav_filename(name):
    """'107s1', '107_s1', '107 s 1', '107_s1.wav' -> '107_s1.wav'; None if not that shape."""
    m = re.fullmatch(r"(\d+)s(\d+)", str(name).lower().replace(".wav", "").replace(" ", "").replace("_", ""))
    return f"{m[1]}_s{m[2]}.wav" if m else None


def recording_id(path):
    """'wav files/107_s3.wav' -> '107' (the recording/student; the CV group)."""
    return os.path.basename(path).split("_s")[0]


def load_pairs(cfg):
    rows = []
    excel_files = sorted(glob.glob(os.path.join(cfg["excel_dir"], "*.xlsx")))
    if not excel_files:
        raise SystemExit(f"No .xlsx files in {cfg['excel_dir']} - set excel_dir in config.yaml")
    missing = 0
    for xf in excel_files:
        base = os.path.basename(xf)
        m = re.search(r"output_(\d+)_s", base.lower())
        df = pd.read_excel(xf)
        df.columns = [str(c).strip().lower() for c in df.columns]
        lbl_col = _pick(df.columns, ["clarity", "label", "labels", "class", "category", "score", "mos"])
        if lbl_col is None:
            log.warning("no label column in %s, skipped", base)
            continue
        fn_col = _pick(df.columns, ["file name", "filename", "file_name", "wav", "wavname", "audio", "path", "clip"])
        if fn_col is not None:
            names = [to_wav_filename(v) for v in df[fn_col]]
        else:  # no filename column: id from the Excel name + a segment-index column
            seg_col = _pick(df.columns, ["s", "s_no", "segment", "seg", "take", "index", "id", "number", "no"])
            if m is None or seg_col is None:
                log.warning("cannot infer filenames in %s, skipped", base)
                continue
            names = [f"{m[1]}_s{int(v)}.wav" if str(v).strip().isdigit() else None for v in df[seg_col]]
        for nm, lbl in zip(names, map(normalize_label, df[lbl_col])):
            if nm is None or lbl is None:
                continue
            p = os.path.join(cfg["wav_dir"], nm)
            if os.path.exists(p):
                rows.append((p, lbl, recording_id(p)))
            else:
                missing += 1
    data = pd.DataFrame(rows, columns=["path", "label", "recording_id"]).drop_duplicates("path")
    log.info("%d Excel files, %d labelled clips matched, %d labelled rows with no wav", len(excel_files), len(data), missing)
    if data.empty:
        raise SystemExit("No matched (wav, label) pairs. Check config.yaml paths.")
    return data.reset_index(drop=True)
