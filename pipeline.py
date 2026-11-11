import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

def make_demo_data(n=2600, seed=42):
    rng=np.random.default_rng(seed)
    stops=np.array(["Central Station","Market Street","University","City Hospital","Airport Road","Tech Park","Old Town"])
    # Intentionally imbalanced boarding classes (class labels 1..5)
    y=rng.choice([1,2,3,4,5], size=n, p=[.62,.23,.10,.04,.01])
    hour=rng.integers(5,24,n)
    weekday=rng.integers(0,7,n)
    stop=rng.choice(stops,n)
    weather=rng.choice(["clear","rain","cloudy"],n,p=[.65,.15,.20])
    # weak signal to keep the demo a genuine prediction task
    hour_effect=np.where((hour>=7)&(hour<=10),1,np.where((hour>=16)&(hour<=19),2,0))
    y=np.clip(y+((hour_effect>0)&(rng.random(n)<.12)).astype(int),1,5)
    return pd.DataFrame({"StopName":stop,"WeekBeginning":pd.Timestamp("2025-01-06")+pd.to_timedelta(weekday,"D")+pd.to_timedelta(hour,"h"),
                         "Hour":hour,"Weekday":weekday,"Weather":weather,"NumberOfBoardings":y})

def validate_dataset(df):
    errors=[]
    if "NumberOfBoardings" not in df.columns: errors.append("Missing required target column: NumberOfBoardings.")
    if len(df)<50: errors.append("Dataset must contain at least 50 rows for a meaningful train/test experiment.")
    if "WeekBeginning" not in df.columns and not any(c.lower()=="hour" for c in df.columns):
        errors.append("Provide WeekBeginning (datetime) or an Hour feature.")
    if "NumberOfBoardings" in df and df["NumberOfBoardings"].nunique()<2:
        errors.append("Target must contain at least two distinct classes.")
    return errors

class TabularGAN:
    """Small conditional tabular GAN trained only on minority-class training rows."""
    def __init__(self, n_features, n_classes, noise_dim=16):
        self.noise_dim=noise_dim
        self.G=nn.Sequential(nn.Linear(noise_dim+n_classes,64),nn.ReLU(),
                             nn.Linear(64,64),nn.ReLU(),nn.Linear(64,n_features),nn.Tanh())
        self.D=nn.Sequential(nn.Linear(n_features+n_classes,64),nn.LeakyReLU(.2),
                             nn.Linear(64,32),nn.LeakyReLU(.2),nn.Linear(32,1),nn.Sigmoid())
    def fit_sample(self, X, labels, per_class=200, epochs=15, seed=42):
        torch.manual_seed(seed); np.random.seed(seed)
        X=np.asarray(X,dtype=np.float32)
        classes, counts=np.unique(labels,return_counts=True)
        target=int(counts.max())
        xs=[]; ys=[]
        opt_g=torch.optim.Adam(self.G.parameters(),lr=0.001)
        opt_d=torch.optim.Adam(self.D.parameters(),lr=0.001)
        bce=nn.BCELoss()
        for cls,count in zip(classes,counts):
            need=min(target-int(count),per_class)
            if need<=0: continue
            real=X[np.asarray(labels)==cls]
            if len(real)<2:
                # GAN cannot learn a distribution from one row; bootstrap that row with tiny noise.
                gen=np.repeat(real,need,axis=0)+np.random.normal(0,.01,(need,X.shape[1])).astype(np.float32)
                xs.append(np.clip(gen,-1,1)); ys.extend([cls]*need); continue
            cond=np.zeros((len(real),len(classes)),dtype=np.float32)
            ci=list(classes).index(cls); cond[:,ci]=1
            xr=torch.tensor(real); cr=torch.tensor(cond)
            for _ in range(max(1,epochs)):
                idx=torch.randint(0,len(xr),(min(64,len(xr)),))
                real_b=xr[idx]; cond_b=cr[idx]
                z=torch.randn(len(idx),self.noise_dim)
                fake=self.G(torch.cat([z,cond_b],1)).detach()
                d_real=self.D(torch.cat([real_b,cond_b],1))
                d_fake=self.D(torch.cat([fake,cond_b],1))
                loss_d=bce(d_real,torch.ones_like(d_real))+bce(d_fake,torch.zeros_like(d_fake))
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()
                z=torch.randn(len(idx),self.noise_dim)
                fake=self.G(torch.cat([z,cond_b],1))
                d_fake=self.D(torch.cat([fake,cond_b],1))
                loss_g=bce(d_fake,torch.ones_like(d_fake))
                opt_g.zero_grad(); loss_g.backward(); opt_g.step()
            cond_new=torch.zeros((need,len(classes))); cond_new[:,ci]=1
            with torch.no_grad():
                gen=self.G(torch.cat([torch.randn(need,self.noise_dim),cond_new],1)).numpy()
            xs.append(np.clip(gen,-1,1)); ys.extend([cls]*need)
        if not xs: return np.empty((0,X.shape[1])), np.array([])
        return np.vstack(xs).astype(np.float32),np.asarray(ys)

class SmallCNN(nn.Module):
    def __init__(self,n_features,n_classes):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,16,(1,3),padding=(0,1)),nn.ReLU(),
                               nn.Conv2d(16,32,(1,3),padding=(0,1)),nn.ReLU(),
                               nn.AdaptiveAvgPool2d((1,4)),nn.Flatten(),
                               nn.Linear(32*4,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,n_classes))
    def forward(self,x): return self.net(x)

def train_cnn(Xtr,ytr,Xte,epochs=15,seed=42):
    torch.manual_seed(seed)
    classes=np.unique(ytr); mapping={v:i for i,v in enumerate(classes)}
    yt=np.array([mapping[v] for v in ytr])
    model=SmallCNN(Xtr.shape[1],len(classes))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    loss_fn=nn.CrossEntropyLoss()
    x=torch.tensor(Xtr,dtype=torch.float32).reshape(-1,1,1,Xtr.shape[1])
    y=torch.tensor(yt,dtype=torch.long)
    model.train()
    for _ in range(epochs):
        order=torch.randperm(len(x))
        for ids in order.split(128):
            logits=model(x[ids]); loss=loss_fn(logits,y[ids])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred=model(torch.tensor(Xte,dtype=torch.float32).reshape(-1,1,1,Xte.shape[1])).argmax(1).numpy()
    inv={i:v for v,i in mapping.items()}
    return np.array([inv[i] for i in pred])

def run_experiment(df,epochs=15,gan_samples=200,test_size=.2,random_state=42):
    data=df.copy()
    data=data.dropna(subset=["NumberOfBoardings"])
    y=data.pop("NumberOfBoardings")
    # Features exactly mirror the paper's date extraction, while retaining other fields.
    if "WeekBeginning" in data:
        dt=pd.to_datetime(data["WeekBeginning"],errors="coerce")
        data["year_value"]=dt.dt.year; data["month"]=dt.dt.month; data["day"]=dt.dt.day
        data["hour"]=dt.dt.hour; data["minute"]=dt.dt.minute; data["second"]=dt.dt.second
        data=data.drop(columns=["WeekBeginning"])
    if "StopName" in data: data["StopName"]=data["StopName"].astype(str)
    # Drop columns that are entirely empty; normalize target labels.
    data=data.replace([np.inf,-np.inf],np.nan)
    for c in data.select_dtypes(include=["object"]).columns: data[c]=data[c].fillna("Unknown").astype(str)
    num=data.select_dtypes(include=np.number).columns.tolist()
    cat=[c for c in data.columns if c not in num]
    transformers=[]
    if num: transformers.append(("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num))
    if cat: transformers.append(("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat))
    prep=ColumnTransformer(transformers,remainder="drop")
    Xtr_raw,Xte_raw,ytr,yte=train_test_split(data,y,test_size=test_size,random_state=random_state,stratify=y)
    Xtr=prep.fit_transform(Xtr_raw).astype(np.float32); Xte=prep.transform(Xte_raw).astype(np.float32)
    # GAN uses tanh output, so scale training features to [-1, 1].
    lo=Xtr.min(axis=0); hi=Xtr.max(axis=0); span=np.where(hi-lo==0,1,hi-lo)
    Xgan=2*(Xtr-lo)/span-1; Xtegan=2*(Xte-lo)/span-1
    gan=TabularGAN(Xgan.shape[1],len(np.unique(ytr)))
    Xsyn,ysyn=gan.fit_sample(Xgan,ytr.to_numpy(),per_class=gan_samples,epochs=epochs,seed=random_state)
    Xaug=np.vstack([Xgan,Xsyn]) if len(Xsyn) else Xgan
    yaug=np.concatenate([ytr.to_numpy(),ysyn]) if len(ysyn) else ytr.to_numpy()
    # SMOTE baseline: applied only to training partition.
    from imblearn.over_sampling import SMOTE
    smote=SMOTE(random_state=random_state,k_neighbors=max(1,min(5,int(pd.Series(ytr).value_counts().min())-1)))
    Xsm,ysm=smote.fit_resample(Xtr,ytr)
    # DNN baselines
    models={
      "DNN (original)":(Xtr,ytr.to_numpy()),
      "SMOTE + DNN":(Xsm,ysm),
      "GAN + DNN":((Xaug+1)/2*(span)+lo, yaug)
    }
    predictions={}
    for name,(xx,yy) in models.items():
        clf=MLPClassifier(hidden_layer_sizes=(128,64),activation="relu",solver="adam",
                          alpha=.0005,batch_size=min(64,len(yy)),max_iter=max(100,epochs*10),
                          early_stopping=True,random_state=random_state)
        clf.fit(xx,yy); predictions[name]=clf.predict(Xte)
    # CNN2D extension, using GAN-augmented training data.
    pred_cnn=train_cnn((Xaug+1)/2*span+lo,yaug,Xte,epochs=epochs,seed=random_state)
    predictions["GAN + CNN2D"]=pred_cnn
    rows=[]; figs={}
    labels=sorted(pd.unique(y))
    for name,pred in predictions.items():
        rows.append({"Model":name,"Accuracy":accuracy_score(yte,pred),
                     "Precision":precision_score(yte,pred,average="macro",zero_division=0),
                     "Recall":recall_score(yte,pred,average="macro",zero_division=0),
                     "F1":f1_score(yte,pred,average="macro",zero_division=0)})
        fig,ax=plt.subplots(figsize=(6,4))
        sns.heatmap(confusion_matrix(yte,pred,labels=labels),annot=True,fmt="d",cmap="Blues",
                    xticklabels=labels,yticklabels=labels,ax=ax)
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        figs[name]=fig
    metrics=pd.DataFrame(rows).sort_values("F1",ascending=False).reset_index(drop=True)
    pred_df=pd.DataFrame({"Actual":np.asarray(yte),"Predicted_GAN_DNN":predictions["GAN + DNN"]})
    counts=pd.Series(yaug).value_counts().sort_index().rename("Training rows after GAN")
    return {"metrics":metrics,"confusion":figs,"predictions":pred_df,
            "original_rows":len(Xtr),"generated_rows":len(Xsyn),"test_rows":len(Xte),
            "class_counts":counts}
