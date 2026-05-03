import torch                                                                                                                                                                                
import torch.nn as nn                                                                                                                                                                       
from torch.utils.data import DataLoader, WeightedRandomSampler                                                                                                                              
from sklearn.metrics import roc_auc_score                                                                                                                                                   
import numpy as np                                                                                                                                                                          
                                                                                                                                                                                            
class EventLSTM(nn.Module):                                                                                                                                                                 
    def __init__(self, input_dim=1283, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()                                                                                                                                                                  
        self.lstm = nn.LSTM(                                                                                                                                                                
            input_size=input_dim,
            hidden_size=hidden_dim,                                                                                                                                                         
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,                                                                                                                                                                
        )
        self.classifier = nn.Sequential(                                                                                                                                                    
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),                                                                                                                                                                      
            nn.Dropout(dropout),
            nn.Linear(64, 1),                                                                                                                                                               
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]  # last layer hidden state                                                                                                                                         
        return self.classifier(out).squeeze(1)                                                                                                                                              
                                                                                                                                                                                            
                                                                                                                                                                                            
def train(sequences, device='cuda', epochs=50, patience=10, batch_size=16, lr=1e-3):
    from event_dataset import EventSequenceDataset, build_cnn_encoder                                                                                                                       
                
    encoder = build_cnn_encoder(device)                                                                                                                                                     
    dataset = EventSequenceDataset(sequences, encoder, device)
                                                                                                                                                                                            
    # Weighted sampler for class imbalance
    labels  = [s['label'] for s in sequences]
    counts  = [labels.count(0), labels.count(1)]                                                                                                                                            
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)                                                                                                    
                
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler)                                                                                                                   
                
    model     = EventLSTM(input_dim=1283).to(device)                                                                                                                                        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()                                                                                                                                                                

    best_auc   = 0                                                                                                                                                                          
    patience_c = 0
                                                                                                                                                                                            
    for epoch in range(epochs):
        model.train()                                                                                                                                                                       
        epoch_loss = 0

        for seqs, targets in loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()                                                                                                                                                           
            preds = model(seqs)
            loss  = criterion(preds, targets)                                                                                                                                               
            loss.backward()                                                                                                                                                                 
            optimizer.step()
            epoch_loss += loss.item()                                                                                                                                                       
                
        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():                                                                                                                                                               
            for seqs, targets in loader:
                seqs = seqs.to(device)                                                                                                                                                      
                preds = model(seqs).cpu().numpy()
                all_preds.extend(preds)                                                                                                                                                     
                all_labels.extend(targets.numpy())
                                                                                                                                                                                            
        auc = roc_auc_score(all_labels, all_preds)                                                                                                                                          
        avg_loss = epoch_loss / len(loader)
        print(f'Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f} — AUC: {auc:.4f}')                                                                                                          
                                                                                                                                                                                            
        if auc > best_auc:
            best_auc = auc                                                                                                                                                                  
            patience_c = 0
            torch.save(model.state_dict(), 'my_basket_tube/models/event_lstm.pt')
            print(f'  -> Saved best model (AUC {best_auc:.4f})')                                                                                                                            
        else:                                                                                                                                                                               
            patience_c += 1                                                                                                                                                                 
            if patience_c >= patience:                                                                                                                                                      
                print(f'Early stopping at epoch {epoch+1}')
                break                                                                                                                                                                       

    print(f'Training complete. Best AUC: {best_auc:.4f}')                                                                                                                                   
    return model

                                                                                                                                                                                            
if __name__ == '__main__':
    from build_event_training_dataset import sequences                                                                                                                                      
    train(sequences)
