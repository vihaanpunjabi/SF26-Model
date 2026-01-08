import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class FireUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(FireUNet, self).__init__()
        # Contracting Path (Encoder) - Gets small to understand context
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        # Expansive Path (Decoder) - Gets big to pinpoint location
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)
        logits = self.outc(x)
        return torch.sigmoid(logits)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class FirePredictor(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=16):
        super(FirePredictor, self).__init__()
        self.conv_lstm = ConvLSTMCell(input_channels, hidden_channels, 3, True)
        self.final_conv = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x, hidden_state=None):
        b, seq_len, _, h, w = x.size()
        
        if hidden_state is None:
            hx = torch.zeros(b, 16, h, w).to(x.device)
            cx = torch.zeros(b, 16, h, w).to(x.device)
        else:
            hx, cx = hidden_state

        for t in range(seq_len):
            hx, cx = self.conv_lstm(x[:, t, :, :, :], (hx, cx))
            
        prediction = torch.sigmoid(self.final_conv(hx))
        return prediction, (hx, cx)


class PhysicsAdjuster(nn.Module):
    def __init__(self):
        super(PhysicsAdjuster, self).__init__()
        
    def forward(self, predicted_map, wind_speed, wind_dir_deg):
        rad = torch.deg2rad(wind_dir_deg)
        
        shift_x = (torch.sin(rad) * wind_speed * 5).int()
        shift_y = (torch.cos(rad) * wind_speed * 5).int() 

        refined_map = torch.roll(predicted_map, shifts=(shift_y.item(), shift_x.item()), dims=(2, 3))
        
        final_output = (predicted_map + refined_map) / 2
        return final_output