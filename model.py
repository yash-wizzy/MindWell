import torch
import torch.nn as nn

MAX_PHQ_SCORE = 24.0

class DepressionModel(nn.Module):
    def __init__(self, audio_dim=39, text_dim=300, facial_dim=132, hidden_dim=128, num_layers=3, dropout=0.3):
        super(DepressionModel, self).__init__()
        
        self.lstm_audio = nn.LSTM(audio_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        
        self.fc_text1 = nn.Linear(text_dim, hidden_dim)
        self.fc_text2 = nn.Linear(hidden_dim, hidden_dim)
        
        
        self.lstm_facial = nn.LSTM(facial_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        
        self.fc_fusion1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc_fusion2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_fusion3 = nn.Linear(hidden_dim // 2, 1)
        
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio, text, facial):
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1) 
        
       
        lstm_out_audio, _ = self.lstm_audio(audio)
        audio_out = lstm_out_audio[:, -1, :]  
        
        
        text_out = self.relu(self.fc_text1(text))
        text_out = self.relu(self.fc_text2(text_out))
        
       
        if facial.dim() == 2:
            facial = facial.unsqueeze(1)  
        

        lstm_out_facial, _ = self.lstm_facial(facial)
        facial_out = lstm_out_facial[:, -1, :] 
        
        fusion = torch.cat((audio_out, text_out, facial_out), dim=1)
        fusion = self.relu(self.fc_fusion1(fusion))
        fusion = self.dropout(fusion)
        fusion = self.relu(self.fc_fusion2(fusion))
        
        output = torch.sigmoid(self.fc_fusion3(fusion)) * MAX_PHQ_SCORE
        return output

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)