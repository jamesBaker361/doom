import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import torch

class FlatImageFolder(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
    

class MovieImageFolder(Dataset):

    def __init__(self,folder,vae,image_processor,lookback:int):
        super().__init__()
        csv_file=os.path.join(folder,"actions.csv")
        self.df=pd.read_csv(csv_file)
        self.posterior_list=[] #images are stored from 0=t_0, 1=t_1
        self.lookback=lookback
        
        for f,file in enumerate( self.df["file"]):
            
            pil_image=Image.open(os.path.join(folder,file))
            pt_image=image_processor.preprocess(pil_image)
            posterior=vae.encode(pt_image).latent_dist
            self.posterior_list.append(posterior)
            if f ==0:
                self.zero_posterior=torch.zeros(posterior.size())
        
    def __len__(self):
        return len(self.posterior_list)
    
    def __getitem__(self, index):
        episode=self.df.iloc[index]["episode"]
        start=index-self.lookback
        output_dict={column:[] for column in self.df.columns}
        output_dict["posterior"]=[]
        for i in range(start,index):
            if i<0 or self.df.iloc[i]["episode"]!=episode:
                for column in self.df.columns:
                    output_dict[column].append(-1)
                output_dict["posterior"].append(self.zero_posterior)
            else:
                for column in self.df.columns:
                    output_dict[column].append(self.df.iloc[i][column])
                output_dict["posterior"].append(self.posterior_list[i])
        return output_dict