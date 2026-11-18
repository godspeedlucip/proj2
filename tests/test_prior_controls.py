import unittest

import numpy as np
import torch
import torch.nn.functional as F

from rstar4d.evaluation.prior_diagnostics import diagnose
from rstar4d.models.prior_fusion import GeometryPriorFusion
from rstar4d.models.sampling_grid import feature_mask


class PriorControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(192)
            self.model = torch.nn.Module()
            self.model.prior_fusion = GeometryPriorFusion('geometry',29).eval()

    def test_observed_candidate_count_includes_all_valid_coarse_sites(self):
        data = torch.zeros(1,1,2,8,12,16)
        domain = torch.zeros(1,8,12,16,dtype=torch.bool)
        domain[:,2:,4:,4:] = True
        target = torch.ones_like(data[:,0],dtype=torch.bool)
        target[:,0,:,4:] = False
        phases = []
        def observe(index, values):
            phases.append(values)
        context = {'geometry':torch.zeros(1,2,14),'prior_fov':domain,'source_fov':target}
        with torch.no_grad():
            normal = self.model.prior_fusion(data,data[:,:,0],context)
            measured = self.model.prior_fusion(data,data[:,:,0],dict(context,prior_observer=observe))
        torch.testing.assert_close(measured,normal,rtol=0,atol=0)
        expected = F.conv3d(feature_mask(domain).float()[:,None],torch.ones(1,1,3,3,3),padding=1).flatten(1)
        for i,seen in enumerate(phases):
            self.assertIn('coarse_count',seen)
            torch.testing.assert_close(seen['coarse_count'],expected.long())
            torch.testing.assert_close(seen['target_valid'],feature_mask(target[:,i]).flatten(1))

    def test_phase_filters_and_reversing_prior_conditions(self):
        from validation.prior_controls import positive_control
        arrays = positive_control(np.zeros((2,14),np.float32))
        report = diagnose(self.model,arrays,'cpu',.1,phase_labels=[2,9])
        self.assertEqual([p['phase_label'] for p in report['phases']],[2,9])
        self.assertGreater(report['points'],0)
        self.assertEqual(report['points'],sum(p['points'] for p in report['phases']))
        for phase in [report,*report['phases']]:
            counts = list(phase['filter_counts'].values())
            self.assertEqual(counts,sorted(counts,reverse=True))
            self.assertEqual(counts[-1],phase['points'])
            self.assertEqual(phase['matched']['relative'],1.)
        reverse = dict(arrays,matched=arrays['mismatched'],mismatched=arrays['matched'])
        reversed_report = diagnose(self.model,reverse,'cpu',.1)
        self.assertEqual(reversed_report['filter_counts'],report['filter_counts'])
        self.assertEqual(reversed_report['delta_rms_difference'],-report['delta_rms_difference'])
        identical = diagnose(self.model,dict(arrays,mismatched=arrays['matched']),'cpu',.1)
        self.assertEqual(identical['delta_rms_difference'],0.)
