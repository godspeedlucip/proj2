import unittest

import torch

from rstar4d.config import resolve_stage2, validate_train
from rstar4d.models import model as models


class CardiacModelTests(unittest.TestCase):
    def test_off_preserves_fp32_loss_and_parameter_gradients_in_shared_window(self):
        from rstar4d.data.context import crop_core
        from rstar4d.training.losses import compute_loss
        torch.set_num_threads(2)
        main = models.create_model(2)
        wrapped = models.create_model(2)
        wrapped.load_state_dict(main.state_dict(),strict=True)
        opts = resolve_stage2(validate_train({'stage2_crop_mode':'core_with_context'}),2)
        models.attach_stage2_extensions(wrapped,opts)
        x = torch.randn(1,2,2,3,13,17)
        context = {'core_bounds':torch.tensor([[[1,3],[1,12],[3,16]]])}
        a = crop_core(models.forward_volume(main,x),context)
        b = crop_core(models.forward_volume(wrapped,x,context=context),context)
        self.assertTrue(torch.equal(a,b))
        expected = torch.randn_like(a)
        valid = torch.ones_like(expected,dtype=torch.bool)
        for values in (a,b):
            compute_loss(values,expected,valid,2,{'kind':'l1'})['total'].backward()
        for (name,p),(other,q) in zip(main.named_parameters(),wrapped.named_parameters()):
            self.assertEqual(name,other)
            self.assertTrue(torch.equal(p.grad,q.grad),name)

    def test_motion_pairs_no_wrap_and_invalid_source(self):
        self.assertTrue(hasattr(models, 'attach_stage2_extensions'))
        from rstar4d.models.motion import AdjacentMotionFusion
        for count in (2,10,11):
            fusion = AdjacentMotionFusion('identity',[8]*3)
            dynamic = torch.rand(1,1,count,2,8,8)
            features = torch.randn(1,64,count,1,2,2)
            result, aux = fusion(dynamic,features,{'source_fov':torch.zeros_like(dynamic[:,0],dtype=torch.bool)},True)
            self.assertEqual(len(aux['pairs']),2*(count-1))
            self.assertTrue((aux['pairs'].diff().abs() == 1).all())
            self.assertEqual(int((aux['pairs'][:,0] == 0).sum()),1)
            self.assertFalse(result.any())

    def test_single_prior_candidate_passes_signed_cosine_to_gate(self):
        self.assertTrue(hasattr(models, 'attach_stage2_extensions'))
        from rstar4d.models.prior_fusion import GeometryPriorFusion, match_candidates
        query = torch.tensor([[[1.],[0.]]])
        valid = torch.ones(1,1,1,1,dtype=torch.bool)
        fusion = GeometryPriorFusion('image_only',16).eval()
        fov = torch.zeros(1,4,8,8,dtype=torch.bool)
        fov[:,0,0,0] = True
        source = torch.randn(1,1,1,4,8,8)
        context = {'source_fov':torch.ones_like(source[:,0],dtype=torch.bool),'prior_fov':fov}
        seen = []
        fusion.gate[0].register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0][:,56].clone()))
        with torch.no_grad():
            fusion.query.weight.zero_()
            fusion.query.bias.zero_()
            fusion.query.bias[0] = 1.
            fusion.key.weight.zero_()
        for cosine in (.9,-.8):
            key = torch.tensor([[[cosine],[(1-cosine**2)**.5]]])
            _, raw, relative, present = match_candidates(query,key,valid,torch.zeros(1,3,dtype=torch.long))
            self.assertTrue(present.item())
            self.assertAlmostEqual(raw.item(),cosine,places=6)
            self.assertEqual(relative.item(),1.)
            with torch.no_grad():
                fusion.key.bias.zero_()
                fusion.key.bias[:2] = key.flatten()
                fusion(source,source[:,:,0],context)
                torch.testing.assert_close(seen[-1],torch.full_like(seen[-1],cosine))
        with torch.no_grad():
            self.assertFalse(fusion(source,source[:,:,0],{**context,'prior_fov':torch.zeros_like(fov)}).any())

    def test_eight_combinations_restore_and_backpropagate(self):
        self.assertTrue(hasattr(models, 'attach_stage2_extensions'))
        import itertools
        torch.set_num_threads(2)
        x = torch.rand(1,2,2,3,9,9)
        ctx = {'source_fov':torch.ones(1,2,3,9,9,dtype=torch.bool),
               'prior_fov':torch.ones(1,3,9,9,dtype=torch.bool),'geometry':torch.rand(1,2,14)}
        for motion,prior,boundary in itertools.product(('off','learned'),('off','geometry'),(False,True)):
            with self.subTest(motion=motion,prior=prior,boundary=boundary):
                resolved = resolve_stage2(validate_train({'extensions':{'motion':{'mode':motion},
                    'prior':{'mode':prior},'boundary':{'enabled':boundary}}}),2)
                model = models.create_model(2)
                baseline_keys = set(model.state_dict())
                models.attach_stage2_extensions(model,resolved)
                pred = models.forward_volume(model,x,context=ctx)
                pred.square().mean().backward()
                self.assertEqual(pred.shape,(1,1,2,3,9,9))
                if motion == prior == 'off':
                    self.assertEqual(set(model.state_dict()),baseline_keys)
                for branch in (model.motion,model.prior_fusion):
                    if branch is not None:
                        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in branch.parameters()))
                restored = models.model_from_checkpoint({'stage':2,'model_profile':'official_v1',
                    'state_dict':model.state_dict(),'stage2_resolved':resolved})
                with torch.no_grad():
                    torch.testing.assert_close(pred, models.forward_volume(restored,x,context=ctx))
