import torch, sys
sys.stdout.reconfigure(encoding='utf-8')
fnames = ['best_gat_forecaster_asymmetric.pth','best_gat_forecaster_holiday.pth','best_gat_forecaster_weekend.pth']
for fname in fnames:
    ck = torch.load(f'data/models/{fname}', map_location='cpu', weights_only=False)
    fc2w = ck['state_dict']['fc2.weight'].shape
    fc2b = ck['state_dict']['fc2.bias'].shape
    lstm_w = ck['state_dict']['lstm.weight_ih_l0'].shape
    in_ch = ck.get('in_channels', 'N/A')
    ph = ck.get('pred_horizon', 'N/A')
    out_ch = ck.get('out_channels', 'N/A')
    print(f'{fname}:')
    print(f'  fc2.weight={fc2w}  fc2.bias={fc2b}')
    print(f'  lstm_input_size={lstm_w}')
    print(f'  in_channels={in_ch}  pred_horizon={ph}  out_channels={out_ch}')
    print()
