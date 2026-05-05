# =============================================================================
# 3. END-TO-END MODEL ARCHITECTURE & SMART RESUME
# =============================================================================
print("\n[*] Downloading pre-trained ResNet50 and building architecture...")

class EmbeddingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.LayerNorm(512), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, EMBEDDING_DIM, bias=False)
        )
        nn.init.orthogonal_(self.net[0].weight)
        nn.init.orthogonal_(self.net[4].weight)
        
    def forward(self, x): return F.normalize(self.net(x), p=2, dim=1)

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50): 
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(min=1e-7)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=embeddings.device).scatter_(1, labels.view(-1, 1).long(), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s

class FullFaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        self.head = EmbeddingHead()
        
        frozen_count, thawed_count = 0, 0
        for i, child in enumerate(self.backbone):
            if i == 7: # Layer 4
                for param in child.parameters():
                    param.requires_grad = True
                    thawed_count += 1
            else:
                for param in child.parameters():
                    param.requires_grad = False
                    frozen_count += 1
        print(f"[*] Backbone structure: {frozen_count} tensors frozen, {thawed_count} tensors thawed (Layer 4).")

    def forward(self, x): 
        with torch.no_grad():
            for i in range(7):
                x = self.backbone[i](x)
        for i in range(7, len(self.backbone)):
            x = self.backbone[i](x)
        return self.head(x)

model = FullFaceNet().to(device)
arcface = ArcFace(EMBEDDING_DIM, NUM_TRAIN_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {'params': model.head.parameters(), 'lr': 3e-4},
    {'params': arcface.parameters(), 'lr': 3e-4},
    {'params': (p for n, p in model.backbone.named_parameters() if p.requires_grad), 'lr': 1e-5}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler('cuda') 

# 🔥 THE BULLETPROOF AUTO-RESUME ENGINE
START_EPOCH = 0
best_val_acc = 0.0

if os.path.exists(FINAL_WEIGHTS):
    print(f"\n[!] Power Failure Recovery: Found Checkpoint at {FINAL_WEIGHTS}!")
    try:
        checkpoint = torch.load(FINAL_WEIGHTS, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        arcface.load_state_dict(checkpoint['arcface_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        
        # Fast-forward the scheduler
        for _ in range(START_EPOCH):
            scheduler.step()
            
        print(f"[*] Successfully resumed from Epoch {START_EPOCH-1} (Best Acc: {best_val_acc:.2f}%)")
    except Exception as e:
        print(f"[!] Warning: Found a file, but it was corrupted or in the old format. Starting fresh. Error: {e}")
else:
    print("\n[*] No checkpoint found. Starting fresh from Epoch 1.")