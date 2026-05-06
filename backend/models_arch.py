import torch



import torch.nn as nn



from facenet_pytorch import InceptionResnetV1







class FullFaceNet(nn.Module):



    def __init__(self, embedding_dim=512):



        super().__init__()



        



                                  



                                                                         



                                                                          



                                                                    



        self.backbone = InceptionResnetV1(pretrained=None, classify=False)



        



                                                          



        self.custom_head = nn.Sequential(



            nn.Flatten(),



            nn.Linear(512, 512),



            nn.LayerNorm(512), 



            nn.PReLU(),



            nn.Linear(512, embedding_dim),



            nn.LayerNorm(embedding_dim)



        )







    def forward(self, x):



        x = self.backbone(x)



        embeddings = self.custom_head(x)



        return embeddings

