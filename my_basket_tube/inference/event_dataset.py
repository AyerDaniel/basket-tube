import torch                                                                                                                                                                                
from torch.utils.data import Dataset                                                                                                                                                        
from torchvision import transforms, models
import cv2                                                                                                                                                                                  

class EventSequenceDataset(Dataset):                                                                                                                                                        
    def __init__(self, sequences, cnn_encoder, device='cuda'):
        self.sequences = sequences                                                                                                                                                          
        self.encoder   = cnn_encoder
        self.device    = device                                                                                                                                                             
        self.transform = transforms.Compose([
            transforms.ToPILImage(),                                                                                                                                                        
            transforms.Resize((224, 224)),
            transforms.ToTensor(),                                                                                                                                                          
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),                                                                                                                                
        ])
                                                                                                                                                                                            
    def __len__(self):
        return len(self.sequences)
                                                                                                                                                                                            
    def __getitem__(self, idx):
        item     = self.sequences[idx]                                                                                                                                                      
        label    = torch.tensor(item['label'], dtype=torch.float32)
        frames   = item['sequence']                                                                                                                                                         
        vectors  = []
                                                                                                                                                                                            
        for (frame_id, box, ball, frame_path, exists) in frames:                                                                                                                            
            x1, y1, x2, y2 = box
            ball_x, ball_y, ball_conf = ball                                                                                                                                                
                                                                                                                                                                                            
            if exists and any(box):
                img = cv2.imread(frame_path)                                                                                                                                                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                                                                                                                  
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:                                                                                                                                                          
                    crop = img  # fallback to full frame                                                                                                                                    
                tensor = self.transform(crop).unsqueeze(0).to(self.device)
                with torch.no_grad():                                                                                                                                                       
                    embedding = self.encoder(tensor).squeeze(0)
            else:                                                                                                                                                                           
                embedding = torch.zeros(self.encoder.output_dim).to(self.device)
                                                                                                                                                                                            
            ball_vec = torch.tensor(
                [ball_x / 1918.0, ball_y / 1078.0, ball_conf],                                                                                                                              
                dtype=torch.float32                                                                                                                                                         
            ).to(self.device)
                                                                                                                                                                                            
            vectors.append(torch.cat([embedding, ball_vec]))

        sequence_tensor = torch.stack(vectors)  # [210, embedding_dim + 3]                                                                                                                  
        return sequence_tensor, label
                                                                                                                                                                                            
                                                                                                                                                                                            
def build_cnn_encoder(device='cuda'):
    backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')                                                                                                                              
    # Remove classifier head — output is 1280-dim
    encoder = torch.nn.Sequential(*list(backbone.children())[:-1],                                                                                                                          
                                    torch.nn.Flatten())                                                                                                                                      
    encoder.output_dim = 1280                                                                                                                                                               
    for param in encoder.parameters():                                                                                                                                                      
        param.requires_grad = False                                                                                                                                                         
    encoder.eval()
    return encoder.to(device)
