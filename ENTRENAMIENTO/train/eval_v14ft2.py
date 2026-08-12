import csv, glob, os, sys
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from train import TubeDataset, build_model
from eval_benchmark_v10 import mp4_a_npy, puntua
ZOOM,THR=0.6,0.1
JUNIO=f"{HERE}/benchmark_paco_junio.csv"
AM=f"{HERE}/am_fp_bench.txt"; AM_DIR="/home/tomis/SBT/DATASETS/reentreno_produccion/am/reentreno/normal"
FC=f"{HERE}/v12_fc_fp_bench.txt"; FC_DIR=f"{HERE}/v12_fc_fp_clips"
JOSE=f"{HERE}/jose_positivos"
TMPA="/tmp/claude-1000/-home-tomis-SBT/6986c294-b197-4501-8ad5-a9e7982ea745/scratchpad/am_fp_npy"
TMPF="/tmp/claude-1000/-home-tomis-SBT/6986c294-b197-4501-8ad5-a9e7982ea745/scratchpad/fc_fp_npy"
MODELS={"v13-prod":"/home/tomis/SBT/DATASETS/runs/videomae_v13_L4/best.pt","v14-ft (desplegado)":"/home/tomis/SBT/DATASETS/runs/videomae_v14_ft/best_bench.pt","v14-ft2":"/home/tomis/SBT/DATASETS/runs/videomae_v14_ft2/best_bench.pt"}
def carga_mp4(txt,d,tmp):
    os.makedirs(tmp,exist_ok=True); perm=set(open(txt).read().split()); items=[]
    for m in sorted(glob.glob(os.path.join(d,"*.mp4"))):
        if os.path.basename(m) not in perm: continue
        n=os.path.join(tmp,os.path.basename(m)+".npy")
        if os.path.exists(n) or mp4_a_npy(m,n): items.append({"npy":n,"clase":"normal","split":"val"})
    return items
junio=[{"npy":r["npy"],"clase":"hurto","split":"val"} for r in csv.DictReader(open(JUNIO))]
am=carga_mp4(AM,AM_DIR,TMPA); fc=carga_mp4(FC,FC_DIR,TMPF)
jose=[{"npy":p,"clase":"hurto","split":"val"} for p in sorted(glob.glob(os.path.join(JOSE,"*.npy")))]
print(f"junio:{len(junio)} am:{len(am)} fc:{len(fc)} jose:{len(jose)}")
for nm,ck in MODELS.items():
    model,mean,std=build_model("videomae")
    sd=torch.load(ck,map_location="cpu"); sd=sd["model"] if isinstance(sd,dict) and "model" in sd else sd
    model.load_state_dict(sd,strict=False); model.to("cuda").eval()
    print(f"\n== {nm} ==")
    pj=puntua(model,TubeDataset(junio,train=False,mean=mean,std=std,zoom=ZOOM),"cuda")
    print(f"  Paco-junio recall: {sum(p>=THR for p in pj)}/{len(pj)}")
    pjo=puntua(model,TubeDataset(jose,train=False,mean=mean,std=std,zoom=ZOOM),"cuda")
    print(f"  JOSE recall: {sum(p>=THR for p in pjo)}/{len(pjo)} | scores: {' '.join(f'{p:.2f}' for p in pjo)}")
    for tag,it in (("AM",am),("FC",fc)):
        ps=puntua(model,TubeDataset(it,train=False,mean=mean,std=std,zoom=ZOOM),"cuda")
        print(f"  {tag} FP: {sum(p>=THR for p in ps)}/{len(ps)} ({100*sum(p>=THR for p in ps)/max(1,len(ps)):.0f}%)")
    del model; torch.cuda.empty_cache()
