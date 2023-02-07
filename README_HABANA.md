# CosmicTagger

## Current Error

```console
  File "/nfs/AI_testbed/homes/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/src/utils/torch/trainer.py", line 699, in forward_pass
    logits_image = self._net(minibatch_data['image'])
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/nfs/AI_testbed/homes/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/src/networks/torch/uresnet2D.py", line 555, in forward
    x = tuple( self.initial_convolution(_x) for _x in x )
  File "/nfs/AI_testbed/homes/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/src/networks/torch/uresnet2D.py", line 555, in <genexpr>
    x = tuple( self.initial_convolution(_x) for _x in x )
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1148, in _call_impl
    result = forward_call(*input, **kwargs)
  File "/nfs/AI_testbed/homes/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/src/networks/torch/uresnet2D.py", line 61, in forward
    out = self.conv(x)
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1148, in _call_impl
    result = forward_call(*input, **kwargs)
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py", line 457, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py", line 453, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: Number of input channels doesn't match weight channels times groups weight_channel = 2input_channel = 38 1 5 5 2 640 1024 1 groups = 1
```
