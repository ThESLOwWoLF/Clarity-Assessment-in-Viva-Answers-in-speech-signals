
import os,re,glob,joblib
import numpy as np
import pandas as pd
import torch,torchaudio
from transformers import Wav2Vec2FeatureExtractor,WavLMModel
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

# Config
EXCEL_DIR="Excels"
WAV_DIR="wav files"
MODEL_DIR="artifacts"
TEST_SIZE=0.2
RANDOM_STATE=42
PROGRESS_EVERY=20
USE_AMP=True  # mixed precision on RTX4050
os.makedirs(MODEL_DIR,exist_ok=True)

# Helpers
def pick_label_col(cols):
    cols=[c.strip().lower() for c in cols]
    for k in ["clarity","label","labels","class","category","score","mos"]:
        if k in cols:return k
    return None

def pick_fname_col(cols):
    cols=[c.strip().lower() for c in cols]
    for k in ["file name","filename","file_name","wav","wavname","audio","path","clip"]:
        if k in cols:return k
    return None

def normalize_label(x):
    if pd.isna(x):return None
    s=str(x).strip().lower()
    # numeric to bins
    try:
        v=float(s)
        if v>=0.66:return "high"
        if v>=0.33:return "medium"
        return "low"
    except:
        pass
    if "high" in s or s=="h":return "high"
    if "med" in s or s=="m":return "medium"
    if "low" in s or s=="l":return "low"
    return None

def to_wav_filename(name):
    # "107s1","107_s1","107 s 1" → "107_s1.wav"
    b=str(name).lower().replace(".wav","").replace(" ","").replace("_","")
    if "s" not in b:return None
    a=b.split("s")
    if len(a)!=2 or not a[0].isdigit() or not a[1].isdigit():return None
    return f"{a[0]}_s{a[1]}.wav"

# Build dataset
rows=[]
excel_files=sorted(glob.glob(os.path.join(EXCEL_DIR,"*.xlsx")))
print("Excels:",len(excel_files))
for xf in excel_files:
    base=os.path.basename(xf)
    # infer base id
    m=re.search(r"output_(\d+)_s",base.lower())
    base_id=m.group(1) if m else None

    df=pd.read_excel(xf)
    df.columns=[c.strip().lower() for c in df.columns]

    lbl_col=pick_label_col(df.columns)
    if lbl_col is None:
        print(f"[WARN] no label column in {base}. Skipping.");continue

    fn_col=pick_fname_col(df.columns)
    names=[]
    if fn_col is not None:
        for v in df[fn_col].tolist():
            names.append(to_wav_filename(v))
    else:
        # try common segment index columns to pair with base_id
        if base_id is None:
            print(f"[WARN] cannot infer names in {base}. Skipping.");continue
        seg_col=None
        for c in ["s","s_no","segment","seg","take","index","id","number","no"]:
            if c in df.columns:seg_col=c;break
        if seg_col is None:
            print(f"[WARN] no filename/index column in {base}. Skipping.");continue
        for v in df[seg_col].tolist():
            try:i=int(str(v).strip())
            except:names.append(None);continue
            names.append(f"{base_id}_s{i}.wav")

    labels=[normalize_label(v) for v in df[lbl_col].tolist()]
    for nm,lbl in zip(names,labels):
        if nm is None or lbl is None:continue
        p=os.path.join(WAV_DIR,nm)
        if os.path.exists(p):rows.append((p,lbl))
        else:print(f"[MISS] {p}")

data=pd.DataFrame(rows,columns=["path","label"])
print("Usable pairs:",len(data))
if len(data)==0:raise SystemExit("No matched pairs. Check columns or filenames.")

# Load Model
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)
feature_extractor=Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model=WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
model.eval()

@torch.no_grad()
def wavlm_embed(path):
    wav,sr=torchaudio.load(path)
    wav=wav.squeeze(0)

    # Force to 16k
    if sr!=16000:
        wav=torchaudio.functional.resample(wav,sr,16000)
    MAX_LEN=16000*8
    if wav.shape[-1]>MAX_LEN:
        wav=wav[:MAX_LEN]
    # Move into model
    inp=feature_extractor(wav,sampling_rate=16000,return_tensors="pt").to(device)
    # Mixed precision for 6GB GPUs
    if device.type=="cuda":
        import torch
        with torch.amp.autocast('cuda'):
            out=model(**inp).last_hidden_state
    else:
        out=model(**inp).last_hidden_state
    # Mean pooling → single embedding
    emb=out.mean(dim=1).squeeze().cpu().numpy()
    return emb


# Extract Embeddings
X=[];Y=[]
for i,(p,lbl) in enumerate(zip(data["path"],data["label"])):
    X.append(wavlm_embed(p));Y.append(lbl)
    if i%PROGRESS_EVERY==0:print(f"Processed {i}/{len(data)}")
X=np.array(X)
Y=np.array(Y)
print("Embeddings:",X.shape)

# Train
le=LabelEncoder()
y=le.fit_transform(Y)

lda=LinearDiscriminantAnalysis(n_components=2)
Xr=lda.fit_transform(X,y)

Xtr,Xte,ytr,yte=train_test_split(Xr,y,test_size=TEST_SIZE,shuffle=True,stratify=y,random_state=RANDOM_STATE)
clf=SVC(kernel="rbf",C=10,gamma="scale",probability=True,random_state=RANDOM_STATE)
clf.fit(Xtr,ytr)

yp=clf.predict(Xte)
print("\n=== REPORT ===")
print(classification_report(yte,yp,target_names=le.classes_))
print("=== CONFUSION MATRIX ===")
print(confusion_matrix(yte,yp))

# Save
joblib.dump(le,os.path.join(MODEL_DIR,"label_encoder.joblib"))
joblib.dump(lda,os.path.join(MODEL_DIR,"lda.joblib"))
joblib.dump(clf,os.path.join(MODEL_DIR,"svm.joblib"))
feature_extractor.save_pretrained(os.path.join(MODEL_DIR,"wavlm_feature_extractor"))
model.save_pretrained(os.path.join(MODEL_DIR,"wavlm_model"))
data.to_csv(os.path.join(MODEL_DIR,"training_pairs.csv"),index=False)
print(f"Saved artifacts→ {MODEL_DIR}")

# Inference
def load_artifacts(model_dir=MODEL_DIR,device_str=None):
    fe=Wav2Vec2FeatureExtractor.from_pretrained(os.path.join(model_dir,"wavlm_feature_extractor"))
    mdl=WavLMModel.from_pretrained(os.path.join(model_dir,"wavlm_model"))
    if device_str is None:
        dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev=torch.device(device_str)
    mdl=mdl.to(dev).eval()
    enc=joblib.load(os.path.join(model_dir,"label_encoder.joblib"))
    lda_=joblib.load(os.path.join(model_dir,"lda.joblib"))
    svm_=joblib.load(os.path.join(model_dir,"svm.joblib"))
    return fe,mdl,enc,lda_,svm_,dev

@torch.no_grad()
def predict_one(audio_path,model_dir=MODEL_DIR):
    fe,mdl,enc,lda_,svm_,dev=load_artifacts(model_dir)
    wav,sr=torchaudio.load(audio_path)
    wav=wav.squeeze(0)
    if sr!=16000:wav=torchaudio.functional.resample(wav,sr,16000)
    inp=fe(wav,sampling_rate=16000,return_tensors="pt").to(dev)
    if dev.type=="cuda":
        from torch.cuda.amp import autocast
        with autocast():
            out=mdl(**inp).last_hidden_state
    else:
        out=mdl(**inp).last_hidden_state
    emb=out.mean(dim=1).squeeze().cpu().numpy().reshape(1,-1)
    xr=lda_.transform(emb)
    proba=svm_.predict_proba(xr)[0]
    pred=enc.inverse_transform([np.argmax(proba)])[0]
    return pred,{cls:float(p) for cls,p in zip(enc.classes_,proba)}

