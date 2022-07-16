import torch
import math

from ...frontend.torch_frontend import ScatteringTorch
from ..core.scattering1d import scattering1d
from ..core.timefrequency_scattering1d import timefrequency_scattering1d
from ..utils import precompute_size_scattering
from ...toolkit import pack_coeffs_jtfs
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase1D
from ..filter_bank_jtfs import _check_runtime_args_jtfs, _handle_args_jtfs


class ScatteringTorch1D(ScatteringTorch, ScatteringBase1D):
    """
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/frontend/
    torch_frontend.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=True,
            oversampling=0, out_type='array', pad_mode='reflect',
            max_pad_factor=2, analytic=False, normalize='l1-energy',
            r_psi=math.sqrt(.5), register_filters=True, backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order, average,
                oversampling, out_type, pad_mode, max_pad_factor, analytic,
                normalize, r_psi, backend)
        ScatteringBase1D._instantiate_backend(self,
                                              'wavespin.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
        if register_filters:
            self.register_filters()

    def register_filters(self):
        """ This function run the filterbank function that
        will create the filters as numpy array, and then, it
        saves those arrays as module's buffers."""
        n = 0
        # prepare for pytorch
        for k in self.phi_f.keys():
            if type(k) != str:
                self.phi_f[k] = torch.from_numpy(
                    self.phi_f[k]).float()
                self.register_buffer('tensor' + str(n), self.phi_f[k])
                n += 1
        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = torch.from_numpy(
                        psi_f[sub_k]).float()
                    self.register_buffer('tensor' + str(n), psi_f[sub_k])
                    n += 1
        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = torch.from_numpy(
                        psi_f[sub_k]).float()
                    self.register_buffer('tensor' + str(n), psi_f[sub_k])
                    n += 1

    def load_filters(self):
        """This function loads filters from the module's buffer """
        buffer_dict = dict(self.named_buffers())
        n = 0

        for k in self.phi_f.keys():
            if type(k) != str:
                self.phi_f[k] = buffer_dict['tensor' + str(n)]
                n += 1

        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = buffer_dict['tensor' + str(n)]
                    n += 1

        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = buffer_dict['tensor' + str(n)]
                    n += 1

    def scattering(self, x):
        # basic checking, should be improved
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        if self.out_type == 'array' and not self.average:
            raise ValueError("out_type=='array' and average==False are mutually "
                             "incompatible. Please set out_type='list'.")

        if not self.out_type in ('array', 'list'):
            raise RuntimeError("The out_type must be one of 'array' or 'list'.")

        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]

        x = x.reshape((-1, 1) + signal_shape)

        self.load_filters()

        # get the arguments before calling the scattering
        # treat the arguments
        if self.average:
            size_scattering = precompute_size_scattering(
                self.J, self.Q, self.T, max_order=self.max_order, detail=True)
        else:
            size_scattering = 0

        # convert to tensor if it isn't already
        if type(x).__module__.split('.')[0] == 'numpy':
            x = torch.from_numpy(x).to(device=self.psi1_f[0][0].device.type)
        device = self.psi1_f[0][0].device.type
        if x.device.type != device:
            x = x.to(device)

        S = scattering1d(x, self.pad_fn, self.backend.unpad, self.backend, self.J,
                         self.log2_T, self.psi1_f, self.psi2_f,
                         self.phi_f, max_order=self.max_order,
                         average=self.average,
                         ind_start=self.ind_start, ind_end=self.ind_end,
                         oversampling=self.oversampling,
                         size_scattering=size_scattering,
                         out_type=self.out_type,
                         average_global=self.average_global)

        if self.out_type == 'array':
                scattering_shape = S.shape[-2:]
                new_shape = batch_shape + scattering_shape

                S = S.reshape(new_shape)
        else:
            for x in S:
                scattering_shape = x['coef'].shape[-1:]
                new_shape = batch_shape + scattering_shape

                x['coef'] = x['coef'].reshape(new_shape)

        return S

ScatteringTorch1D._document()


class TimeFrequencyScatteringTorch1D(TimeFrequencyScatteringBase1D,
                                     ScatteringTorch1D):
    def __init__(self, J, shape, Q, J_fr=None, Q_fr=2, T=None, F=None,
                 implementation=None, average=True, average_fr=False,
                 oversampling=0, oversampling_fr=None, aligned=True,
                 F_kind='gauss', sampling_filters_fr=('exclude', 'resample'),
                 out_type="array", out_3D=False, max_noncqt_fr=None,
                 out_exclude=None, paths_exclude=None, pad_mode='reflect',
                 pad_mode_fr='conj-reflect-zero', max_pad_factor=2,
                 max_pad_factor_fr=None, analytic=True,
                 normalize='l1-energy', r_psi=math.sqrt(.5),
                 backend="torch"):
        (oversampling_fr, normalize_tm, normalize_fr, r_psi_tm, r_psi_fr,
         max_order_tm, scattering_out_type) = (
            _handle_args_jtfs(oversampling, oversampling_fr, normalize, r_psi,
                              out_type))

        # Second-order scattering object for the time variable
        ScatteringTorch1D.__init__(
            self, J, shape, Q, T, max_order_tm, average, oversampling,
            scattering_out_type, pad_mode, max_pad_factor, analytic,
            normalize_tm, r_psi_tm, backend=backend, register_filters=False)

        # Frequential scattering object
        TimeFrequencyScatteringBase1D.__init__(
            self, J_fr, Q_fr, F, implementation, average_fr, aligned,
            F_kind, sampling_filters_fr, max_pad_factor_fr, pad_mode_fr,
            normalize_fr, r_psi_fr, oversampling_fr, out_3D, max_noncqt_fr,
            out_type, out_exclude, paths_exclude)
        TimeFrequencyScatteringBase1D.build(self)

        self.register_filters()

    def register_filters(self):
        """ This function run the filterbank function that
        will create the filters as numpy array, and then, it
        saves those arrays as module's buffers."""
        n_final = self._register_filters(self, ('phi_f', 'psi1_f', 'psi2_f'))
        # register filters from freq-scattering object (see base_frontend.py)
        self._register_filters(self.scf,
                               ('phi_f_fr', 'psi1_f_fr_up', 'psi1_f_fr_dn'),
                               n0=n_final)

    def _register_filters(self, obj, filter_names, n0=0):
        n = n0
        for name in filter_names:
            p_f = getattr(obj, name)
            if name.startswith('psi') and 'fr' not in name:
                for n_tm in range(len(p_f)):
                    for k in p_f[n_tm]:
                        if not isinstance(k, int):
                            continue
                        p_f[n_tm][k] = torch.from_numpy(p_f[n_tm][k]).float()
                        self.register_buffer(f'tensor{n}', p_f[n_tm][k])
                        n += 1
            elif name.startswith('psi') and 'fr' in name:
                for psi_id in p_f:
                    if not isinstance(psi_id, int):
                        continue
                    for n1_fr in range(len(p_f[psi_id])):
                        p_f[psi_id][n1_fr] = torch.from_numpy(
                            p_f[psi_id][n1_fr]).float()
                        self.register_buffer(f'tensor{n}', p_f[psi_id][n1_fr])
                        n += 1
            elif name == 'phi_f':
                for trim_tm in p_f:
                    if not isinstance(trim_tm, int):
                        continue
                    for k in range(len(p_f[trim_tm])):
                        p_f[trim_tm][k] = torch.from_numpy(p_f[trim_tm][k]
                                                           ).float()
                        self.register_buffer(f'tensor{n}', p_f[trim_tm][k])
                        n += 1
            elif name == 'phi_f_fr':
                for log2_F_phi_diff in p_f:
                    if not isinstance(log2_F_phi_diff, int):
                        continue
                    for pad_diff in p_f[log2_F_phi_diff]:
                        for sub in range(len(p_f[log2_F_phi_diff][pad_diff])):
                            p_f[log2_F_phi_diff][pad_diff][sub] = (
                                torch.from_numpy(
                                    p_f[log2_F_phi_diff][pad_diff][sub]).float())
                            self.register_buffer(
                                f'tensor{n}', p_f[log2_F_phi_diff][pad_diff][sub])
                            n += 1
            else:
                raise ValueError("unknown filter name: %s" % name)

        n_final = n
        return n_final

    def load_filters(self):
        """This function loads filters from the module's buffer """
        n_final = self._load_filters(self, ('phi_f', 'psi1_f', 'psi2_f'))
        # register filters from freq-scattering object (see base_frontend.py)
        self._load_filters(self.scf,
                           ('phi_f_fr', 'psi1_f_fr_up', 'psi1_f_fr_dn'),
                           n0=n_final)

    def _load_filters(self, obj, filter_names, n0=0):
        buffer_dict = dict(self.named_buffers())
        n = n0
        for name in filter_names:
            p_f = getattr(obj, name)
            if name.startswith('psi') and 'fr' not in name:
                for n_tm in range(len(p_f)):
                    for k in p_f[n_tm]:
                        if not isinstance(k, int):
                            continue
                        p_f[n_tm][k] = buffer_dict[f'tensor{n}']
                        n += 1
            elif name.startswith('psi') and 'fr' in name:
                for psi_id in p_f:
                    if not isinstance(psi_id, int):
                        continue
                    for n1_fr in range(len(p_f[psi_id])):
                        p_f[psi_id][n1_fr] = buffer_dict[f'tensor{n}']
                        n += 1
            elif name == 'phi_f':
                for trim_tm in p_f:
                    if not isinstance(trim_tm, int):
                        continue
                    for k in range(len(p_f[trim_tm])):
                        p_f[trim_tm][k] = buffer_dict[f'tensor{n}']
                        n += 1
            elif name == 'phi_f_fr':
                for log2_F_phi_diff in p_f:
                    if not isinstance(log2_F_phi_diff, int):
                        continue
                    for pad_diff in p_f[log2_F_phi_diff]:
                        for sub in range(len(p_f[log2_F_phi_diff][pad_diff])):
                            p_f[log2_F_phi_diff
                                ][pad_diff][sub] = buffer_dict[f'tensor{n}']
                            n += 1
            else:
                raise ValueError("unknown filter name: %s" % name)

        n_final = n
        return n_final

    def scattering(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        _check_runtime_args_jtfs(self.average, self.average_fr, self.out_type,
                                 self.out_3D)

        signal_shape = x.shape[-1:]
        x = x.reshape((-1, 1) + signal_shape)

        self.load_filters()

        # convert to tensor if it isn't already, and move to appropriate device
        if type(x).__module__.split('.')[0] == 'numpy':
            x = torch.from_numpy(x)
        device = self.psi1_f[0][0].device.type
        if x.device != device:
            x = x.to(device)

        S = timefrequency_scattering1d(
            x,
            self.backend.pad, self.backend.unpad,
            self.backend,
            self.J,
            self.log2_T,
            self.psi1_f, self.psi2_f, self.phi_f,
            self.scf,
            self.pad_fn,
            average=self.average,
            average_global=self.average_global,
            average_global_phi=self.average_global_phi,
            pad_left=self.pad_left, pad_right=self.pad_right,
            ind_start=self.ind_start, ind_end=self.ind_end,
            oversampling=self.oversampling,
            oversampling_fr=self.oversampling_fr,
            aligned=self.aligned,
            F_kind=self.F_kind,
            out_type=self.out_type,
            out_3D=self.out_3D,
            out_exclude=self.out_exclude,
            paths_exclude=self.paths_exclude,
            pad_mode=self.pad_mode)
        if self.out_structure is not None:
            S = pack_coeffs_jtfs(S, self.meta(), self.out_structure,
                                 separate_lowpass=True,
                                 sampling_psi_fr=self.sampling_psi_fr)
        return S

    def scf_compute_padding_fr(self):
        raise NotImplementedError("Here for docs; implemented in "
                                  "`_FrequencyScatteringBase`.")

    def scf_compute_J_pad_fr(self):
        raise NotImplementedError("Here for docs; implemented in "
                                  "`_FrequencyScatteringBase`.")

TimeFrequencyScatteringTorch1D._document()


__all__ = ['ScatteringTorch1D', 'TimeFrequencyScatteringTorch1D']
