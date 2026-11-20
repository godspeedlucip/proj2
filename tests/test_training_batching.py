import unittest
from unittest.mock import patch

import torch

from rstar4d.config import resolve_stage2, train_defaults, validate_train
from rstar4d.training import trainer


class TrainingBatchingTests(unittest.TestCase):
    def test_group_two_compatible_draws_and_keep_incompatible_tail(self):
        draws = [{'input':torch.zeros(1,2,t,1,8,8)} for t in (10,10,11,10,11)]
        grouped = list(trainer._microbatches(draws,2))
        self.assertEqual([[index for index,_ in batch] for batch in grouped], [[0,1],[2],[3],[4]])

    def test_two_blocks_share_forward_with_separate_core_losses(self):
        cfg = train_defaults()
        resolved = resolve_stage2(cfg,2)
        loss_cfg = {**cfg['loss'],'kind':'l1'}
        model = torch.nn.Conv3d(2,1,1,bias=False)
        device = torch.device('cpu')
        cases = []
        for start in (0,1):
            x = torch.randn(1,2,2,4,8,8)
            cases.append({'input':x,'target':torch.zeros_like(x[:,:1]),
                          'valid':torch.ones_like(x[:,:1],dtype=torch.bool),
                          'context':{'core_bounds':torch.tensor([[[start,start+2],[0,8],[0,8]]])}})

        def forward(net, image, *, context):
            return torch.stack([net(image[:,:,t]) for t in range(image.shape[2])],dim=2)

        with patch.object(trainer,'forward_volume',side_effect=forward) as observed:
            batched = trainer._compute_microbatch(model,cases,device,2,resolved,loss_cfg,1,False)
            self.assertEqual(observed.call_args.args[1].shape[0],2)
            self.assertEqual(observed.call_count,1)
            separate = [trainer.compute_batch(model,case,device,2,resolved,loss_cfg,1,False) for case in cases]

        def combined_grad(results):
            mean = torch.stack([loss['total'] for loss,_ in results]).mean()
            return torch.autograd.grad(mean,model.weight)[0]

        torch.testing.assert_close(combined_grad(batched),combined_grad(separate))

    def test_physical_batch_must_divide_effective_batch(self):
        self.assertEqual(validate_train({'micro_batch_size':2,'accumulation':2})['micro_batch_size'],2)
        with self.assertRaises(ValueError):
            validate_train({'micro_batch_size':2,'accumulation':3})


if __name__ == '__main__':
    unittest.main()
