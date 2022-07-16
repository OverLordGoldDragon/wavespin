import numpy as np
import math
import warnings
from scipy.fft import ifft

def adaptive_choice_P(sigma, eps=1e-7):
    """
    Adaptive choice of the value of the number of periods in the frequency
    domain used to compute the Fourier transform of a Morlet wavelet.

    This function considers a Morlet wavelet defined as the sum
    of
    * a Gabor term hat psi(omega) = hat g_{sigma}(omega - xi)
    where 0 < xi < 1 is some frequency and g_{sigma} is
    the Gaussian window defined in Fourier by
    hat g_{sigma}(omega) = e^{-omega^2/(2 sigma^2)}
    * a low pass term \\hat \\phi which is proportional to \\hat g_{\\sigma}.

    If \\sigma is too large, then these formula will lead to discontinuities
    in the frequency interval [0, 1] (which is the interval used by numpy.fft).
    We therefore choose a larger integer P >= 1 such that at the boundaries
    of the Fourier transform of both filters on the interval [1-P, P], the
    magnitude of the entries is below the required machine precision.
    Mathematically, this means we would need P to satisfy the relations:

    |\\hat \\psi(P)| <= eps and |\\hat \\phi(1-P)| <= eps

    Since 0 <= xi <= 1, the latter implies the former. Hence the formula which
    is easily derived using the explicit formula for g_{\\sigma} in Fourier.

    Parameters
    ----------
    sigma: float
        Positive number controlling the bandwidth of the filters
    eps : float, optional
        Positive number containing required precision. Defaults to 1e-7

    Returns
    -------
    P : int
        integer controlling the number of periods used to ensure the
        periodicity of the final Morlet filter in the frequency interval
        [0, 1[. The value of P will lead to the use of the frequency
        interval [1-P, P[, so that there are 2*P - 1 periods.

    References
    ----------
    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    filter_bank.py
    """
    val = math.sqrt(-2 * (sigma**2) * math.log(eps))
    P = int(math.ceil(val + 1))
    return P


def periodize_filter_fourier(h_f, nperiods=1, aggregation='sum'):
    """
    Computes a periodization of a filter provided in the Fourier domain.
    Parameters
    ----------
    h_f : array_like
        complex numpy array of shape (N*n_periods,)
    n_periods: int, optional
        Number of periods which should be used to periodize
    aggregation: str['sum', 'mean'], optional
        'sum' will multiply subsampled time-domain signal by subsampling
        factor to conserve energy during scattering (rather not double-account
        for it since we already subsample after convolving).
        'mean' will only subsample the input.

    Returns
    -------
    v_f : array_like
        complex numpy array of size (N,), which is a periodization of
        h_f as described in the formula:
        v_f[k] = sum_{i=0}^{n_periods - 1} h_f[i * N + k]

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    filter_bank.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    N = h_f.shape[0] // nperiods
    h_f_re = h_f.reshape(nperiods, N)
    v_f = (h_f_re.sum(axis=0) if aggregation == 'sum' else
           h_f_re.mean(axis=0))
    v_f = v_f if h_f.ndim == 1 else v_f[:, None]  # preserve dim
    return v_f


def morlet_1d(N, xi, sigma, normalize='l1', P_max=5, eps=1e-7):
    """
    Computes the Fourier transform of a Morlet filter.

    A Morlet filter is the sum of a Gabor filter and a low-pass filter
    to ensure that the sum has exactly zero mean in the temporal domain.
    It is defined by the following formula in time:
    psi(t) = g_{sigma}(t) (e^{i xi t} - beta)
    where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
    the cancelling parameter.

    Parameters
    ----------
    N : int
        size of the temporal support
    xi : float
        central frequency (in [0, 1])
    sigma : float
        bandwidth parameter
    normalize : string, optional
        normalization types for the filters. Defaults to 'l1'.
        Supported normalizations are 'l1' and 'l2' (understood in time domain).
    P_max: int, optional
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most 2*P_max - 1 periods
        are used, to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float
        required machine precision (to choose the adequate P)

    Returns
    -------
    morlet_f : array_like
        numpy array of size (N,) containing the Fourier transform of the Morlet
        filter at the frequencies given by np.fft.fftfreq(N).

    References
    ----------
    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    filter_bank.py
    """
    if type(P_max) != int:
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    # Find the adequate value of P
    P = min(adaptive_choice_P(sigma, eps=eps), P_max)
    assert P >= 1
    # Define the frequencies over [1-P, P[
    freqs = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    if P == 1:
        # in this case, make sure that there is continuity around 0
        # by using the interval [-0.5, 0.5]
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = freqs
    # define the gabor at freq xi and the low-pass, both of width sigma
    gabor_f = np.exp(-(freqs - xi)**2 / (2 * sigma**2))
    low_pass_f = np.exp(-(freqs_low**2) / (2 * sigma**2))
    # discretize in signal <=> periodize in Fourier
    gabor_f = periodize_filter_fourier(gabor_f, nperiods=2 * P - 1)
    low_pass_f = periodize_filter_fourier(low_pass_f, nperiods=2 * P - 1)
    # find the summation factor to ensure that morlet_f[0] = 0.
    kappa = gabor_f[0] / low_pass_f[0]
    morlet_f = gabor_f - kappa * low_pass_f
    # normalize the Morlet if necessary
    morlet_f *= get_normalizing_factor(morlet_f, normalize=normalize)
    return morlet_f


def get_normalizing_factor(h_f, normalize='l1'):
    """
    Computes the desired normalization factor for a filter defined in Fourier.

    Parameters
    ----------
    h_f : array_like
        numpy vector containing the Fourier transform of a filter
    normalized : string, optional
        desired normalization type, either 'l1' or 'l2'. Defaults to 'l1'.

    Returns
    -------
    norm_factor : float
        such that h_f * norm_factor is the adequately normalized vector.

    References
    ----------
    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    filter_bank.py
    """
    h_real = ifft(h_f)
    if np.abs(h_real).sum() < 1e-7:
        raise ValueError('Zero division error is very likely to occur, ' +
                         'aborting computations now.')
    normalize = normalize.split('-')[0]  # in case of `-energy`
    if normalize == 'l1':
        norm_factor = 1. / (np.abs(h_real).sum())
    elif normalize == 'l2':
        norm_factor = 1. / np.sqrt((np.abs(h_real)**2).sum())
    else:
        raise ValueError("Supported normalizations only include 'l1' and 'l2'")
    return norm_factor


def gauss_1d(N, sigma, normalize='l1', P_max=5, eps=1e-7):
    """
    Computes the Fourier transform of a low pass gaussian window.

    \\hat g_{\\sigma}(\\omega) = e^{-\\omega^2 / 2 \\sigma^2}

    Parameters
    ----------
    N : int
        size of the temporal support
    sigma : float
        bandwidth parameter
    normalize : string, optional
        normalization types for the filters. Defaults to 'l1'
        Supported normalizations are 'l1' and 'l2' (understood in time domain).
    P_max : int, optional
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most 2*P_max - 1 periods
        are used, to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float, optional
        required machine precision (to choose the adequate P)

    Returns
    -------
    g_f : array_like
        numpy array of size (N,) containing the Fourier transform of the
        filter (with the frequencies in the np.fft.fftfreq convention).

    References
    ----------
    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    filter_bank.py
    """
    # Find the adequate value of P
    if type(P_max) != int:
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    P = min(adaptive_choice_P(sigma, eps=eps), P_max)
    assert P >= 1
    # switch cases
    if P == 1:
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    # define the low pass
    g_f = np.exp(-freqs_low**2 / (2 * sigma**2))
    # periodize it
    g_f = periodize_filter_fourier(g_f, nperiods=2 * P - 1)
    # normalize the signal
    g_f *= get_normalizing_factor(g_f, normalize=normalize)
    # return the Fourier transform
    return g_f


def compute_sigma_psi(xi, Q, r=math.sqrt(0.5)):
    """
    Computes the frequential width sigma for a Morlet filter of frequency xi
    belonging to a family with Q wavelets.

    The frequential width is adapted so that the intersection of the
    frequency responses of the next filter occurs at a r-bandwidth specified
    by r, to ensure a correct coverage of the whole frequency axis.

    Parameters
    ----------
    xi : float
        frequency of the filter in [0, 1]
    Q : int
        number of filters per octave, Q is an integer >= 1
    r : float, optional
        Positive parameter defining the bandwidth to use.
        Should be < 1. We recommend keeping the default value.
        The larger r, the larger the filters in frequency domain.

    Returns
    -------
    sigma : float
        frequential width of the Morlet wavelet.

    References
    ----------
      1. Convolutional operators in the time-frequency domain, V. Lostanlen,
         PhD Thesis, 2017
         https://tel.archives-ouvertes.fr/tel-01559667
      2. Kymatio, (C) 2018-present. The Kymatio developers.
         https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
         filter_bank.py
    """
    factor = 1. / math.pow(2, 1. / Q)
    term1 = (1 - factor) / (1 + factor)
    term2 = 1. / math.sqrt(2 * math.log(1. / r))
    return xi * term1 * term2


def compute_temporal_support(h_f, criterion_amplitude=1e-3, warn=False):
    """
    Computes the (half) temporal support of a family of centered,
    symmetric filters h provided in the Fourier domain

    This function computes the support N which is the smallest integer
    such that for all signals x and all filters h,

    \\| x \\conv h - x \\conv h_{[-N, N]} \\|_{\\infty} \\leq \\epsilon
        \\| x \\|_{\\infty}  (1)

    where 0<\\epsilon<1 is an acceptable error, and h_{[-N, N]} denotes the
    filter h whose support is restricted in the interval [-N, N]

    The resulting value N used to pad the signals to avoid boundary effects
    and numerical errors.

    If the support is too small, no such N might exist.
    In this case, N is defined as the half of the support of h, and a
    UserWarning is raised.

    Parameters
    ----------
    h_f : array_like
        a numpy array of size batch x time, where each row contains the
        Fourier transform of a filter which is centered and whose absolute
        value is symmetric
    criterion_amplitude : float, optional
        value \\epsilon controlling the numerical
        error. The larger criterion_amplitude, the smaller the temporal
        support and the larger the numerical error. Defaults to 1e-3
    warn: bool (default False)
        Whether to raise a warning upon `h_f` leading to boundary effects.

    Returns
    -------
    t_max : int
        temporal support which ensures (1) for all rows of h_f

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    filter_bank.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    if h_f.ndim == 2 and h_f.shape[0] > h_f.shape[1]:
        h_f = h_f.transpose(1, 0)
    elif h_f.ndim == 1:
        h_f = h_f[None]
    elif h_f.ndim > 2:
        raise ValueError("`h_f.ndim` must be <=2, got shape %s" % str(h_f.shape))
    if h_f.shape[-1] == 1:
        return 1

    h = ifft(h_f, axis=1)
    half_support = h.shape[1] // 2
    # check if any value in half of worst case of abs(h) is below criterion
    hhalf = np.max(np.abs(h[:, :half_support]), axis=0)
    max_amplitude = hhalf.max()
    meets_criterion_idxs = np.where(hhalf <= criterion_amplitude * max_amplitude
                                    )[0]
    if len(meets_criterion_idxs) != 0:
        # if it is possible
        N = meets_criterion_idxs.min() + 1
        # in this case pretend it's 1 less so external computations don't
        # have to double support since this is close enough
        if N == half_support:
            N -= 1
    else:
        # if there are none
        N = half_support
        if warn:
            # Raise a warning to say that there will be border effects
            warnings.warn('Signal support is too small to avoid border effects')
    return N


def compute_minimum_required_length(fn, N_init, max_N=None,
                                    criterion_amplitude=1e-3):
    """Computes minimum required number of samples for `fn(N)` to have temporal
    support less than `N`, as determined by `compute_temporal_support`.

    Parameters
    ----------
    fn: FunctionType
        Function / lambda taking `N` as input and returning a filter in
        frequency domain.
    N_init: int
        Initial input to `fn`, will keep doubling until `N == max_N` or
        temporal support of `fn` is `< N`.
    max_N: int / None
        See `N_init`; if None, will raise `N` indefinitely.
    criterion_amplitude : float, optional
        value \\epsilon controlling the numerical
        error. The larger criterion_amplitude, the smaller the temporal
        support and the larger the numerical error. Defaults to 1e-3

    Returns
    -------
    N: int
        Minimum required number of samples for `fn(N)` to have temporal
        support less than `N`.
    """
    N = 2**math.ceil(math.log2(N_init))  # ensure pow 2
    while True:
        try:
            p_fr = fn(N)
        except ValueError as e:  # get_normalizing_factor()
            if "division" not in str(e):
                raise e
            N *= 2
            continue

        p_halfsupport = compute_temporal_support(
            p_fr, criterion_amplitude=criterion_amplitude, warn=False)

        if N > 1e9:  # avoid crash
            raise Exception("couldn't satisfy stop criterion before `N > 1e9`; "
                            "check `fn`")
        if 2 * p_halfsupport < N or (max_N is not None and N > max_N):
            break
        N *= 2
    return N


def get_max_dyadic_subsampling(xi, sigma, alpha=4.):
    """
    Computes the maximal dyadic subsampling which is possible for a Gabor
    filter of frequency xi and width sigma

    Finds the maximal integer j such that:
    omega_0 < 2^{-(j + 1)}
    where omega_0 is the boundary of the filter, defined as
    omega_0 = xi + alpha * sigma

    This ensures that the filter can be subsampled by a factor 2^j without
    aliasing.

    We use the same formula for Gabor and Morlet filters.

    Parameters
    ----------
    xi : float
        frequency of the filter in [0, 1]
    sigma : float
        frequential width of the filter
    alpha : float, optional
        parameter controlling the error done in the aliasing.
        The larger alpha, the smaller the error. Defaults to 4.

    Returns
    -------
    j : int
        integer such that 2^j is the maximal subsampling accepted by the
        Gabor filter without aliasing.

    References
    ----------
      1. Convolutional operators in the time-frequency domain, V. Lostanlen,
         PhD Thesis, 2017
         https://tel.archives-ouvertes.fr/tel-01559667
      2. Kymatio, (C) 2018-present. The Kymatio developers.
         https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
         filter_bank.py
    """
    upper_bound = min(xi + alpha * sigma, 0.5)
    j = math.floor(-math.log2(upper_bound)) - 1
    j = int(j)
    return j


def move_one_dyadic_step(cv, Q, alpha=4.):
    """
    Computes the parameters of the next wavelet on the low frequency side,
    based on the parameters of the current wavelet.

    This function is used in the loop defining all the filters, starting
    at the wavelet frequency and then going to the low frequencies by
    dyadic steps. This makes the loop in compute_params_filterbank much
    simpler to read.

    The steps are defined as:
    xi_{n+1} = 2^{-1/Q} xi_n
    sigma_{n+1} = 2^{-1/Q} sigma_n

    Parameters
    ----------
    cv : dictionary
        stands for current_value. Is a dictionary with keys:
        *'key': a tuple (j, n) where n is a counter and j is the maximal
            dyadic subsampling accepted by this wavelet.
        *'xi': central frequency of the wavelet
        *'sigma': width of the wavelet
    Q : int
        number of wavelets per octave. Controls the relationship between
        the frequency and width of the current wavelet and the next wavelet.
    alpha : float, optional
        tolerance parameter for the aliasing. The larger alpha,
        the more conservative the algorithm is. Defaults to 4.

    Returns
    -------
    new_cv : dictionary
        a dictionary with the same keys as the ones listed for cv,
        whose values are updated

    References
    ----------
    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    filter_bank.py
    """
    factor = 1. / math.pow(2., 1. / Q)
    n = cv['key']
    new_cv = {'xi': cv['xi'] * factor, 'sigma': cv['sigma'] * factor}
    # compute the new j
    new_cv['j'] = get_max_dyadic_subsampling(new_cv['xi'], new_cv['sigma'],
                                             alpha=alpha)
    new_cv['key'] = n + 1
    return new_cv


def compute_xi_max(Q):
    """
    Computes the maximal xi to use for the Morlet family, depending on Q.

    Parameters
    ----------
    Q : int
        number of wavelets per octave (integer >= 1)

    Returns
    -------
    xi_max : float
        largest frequency of the wavelet frame.

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    filter_bank.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    xi_max = max(1. / (1. + math.pow(2., 1. / Q)), 0.4)
    return xi_max


def compute_params_filterbank(sigma_min, Q, r_psi=math.sqrt(0.5), alpha=4.,
                              J_pad=None):
    """
    Computes the parameters of a Morlet wavelet filterbank.

    This family is defined by constant ratios between the frequencies and
    width of adjacent filters, up to a minimum frequency where the frequencies
    are translated. sigma_min specifies the smallest frequential width
    among all filters, while preserving the coverage of the whole frequency
    axis.

    The keys of the dictionaries are tuples of integers (j, n) where n is a
    counter (starting at 0 for the highest frequency filter) and j is the
    maximal dyadic subsampling accepted by this filter.

    Parameters
    ----------
    sigma_min : float
        This acts as a lower-bound on the frequential widths of the band-pass
        filters. The low-pass filter may be wider (if T < 2**J_scattering), making
        invariants over shorter time scales than longest band-pass filter.
    Q : int
        number of wavelets per octave.
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets),
        and stability against time-warp deformations (larger r_psi improves it).
        Defaults to sqrt(0.5).
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 4.
    J_pad : int, optional
        Used to compute `xi_min`, lower bound on `xi` to ensure every bandpass is
        a valid wavelet, i.e. doesn't peak below FFT(x_pad) bin 1. Else, we have

          - extreme distortion (can't peak between 0 and 1)
          - general padding may *introduce* new variability (frequencies) or
            cancel existing ones (e.g. half cycle -> full cycle), so we set
            `xi_min` w.r.t. padded rather than original input

    Returns
    -------
    xi : list[float]
        list containing the central frequencies of the wavelets.
    sigma : list[float]
        list containing the frequential widths of the wavelets.
    j : list[int]
        list containing the subsampling factors of the wavelets (closely
        related to their dyadic scales)
    is_cqt : list[bool]
        list containing True if a wavelet was built per Constant Q Transform
        (fixed `xi / sigma`), else False for the STFT portion

    References
    ----------
      1. Convolutional operators in the time-frequency domain, V. Lostanlen,
         PhD Thesis, 2017
         https://tel.archives-ouvertes.fr/tel-01559667
      2. This is a modification of
         https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
         filter_bank.py
         Kymatio, (C) 2018-present. The Kymatio developers.
    """
    # xi_min
    if J_pad is not None:
        # lowest peak at padded's bin 1
        xi_min = 1 / 2**J_pad
    else:
        # no limit
        xi_min = -1

    xi_max = compute_xi_max(Q)
    sigma_max = compute_sigma_psi(xi_max, Q, r=r_psi)

    xi = []
    sigma = []
    j = []
    is_cqt = []

    if sigma_max <= sigma_min or xi_max <= xi_min:
        # in this exceptional case, we will not go through the loop, so
        # we directly assign
        last_xi = sigma_max
    else:
        # fill all the dyadic wavelets as long as possible
        current = {'key': 0, 'j': 0, 'xi': xi_max, 'sigma': sigma_max}
        # while we can attribute something
        while current['sigma'] > sigma_min and current['xi'] > xi_min:
            xi.append(current['xi'])
            sigma.append(current['sigma'])
            j.append(current['j'])
            is_cqt.append(True)
            current = move_one_dyadic_step(current, Q, alpha=alpha)
        # get the last key
        last_xi = xi[-1]

    # fill num_interm wavelets between last_xi and 0, both excluded
    num_intermediate = Q
    for q in range(1, num_intermediate + 1):
        factor = (num_intermediate + 1. - q) / (num_intermediate + 1.)
        new_xi = factor * last_xi
        new_sigma = sigma_min

        xi.append(new_xi)
        sigma.append(new_sigma)
        j.append(get_max_dyadic_subsampling(new_xi, new_sigma, alpha=alpha))
        is_cqt.append(False)
        if new_xi < xi_min:
            # break after appending one to guarantee tiling `xi_min`
            # in case `xi` increments are too small
            break
    # return results
    return xi, sigma, j, is_cqt


def calibrate_scattering_filters(J, Q, T, r_psi=math.sqrt(0.5), sigma0=0.1,
                                 alpha=4., J_pad=None):
    """
    Calibrates the parameters of the filters used at the 1st and 2nd orders
    of the scattering transform.

    These filterbanks share the same low-pass filterbank, but use a
    different Q: Q_1 = Q and Q_2 = 1.

    The dictionaries for the band-pass filters have keys which are 2-tuples
    of the type (j, n), where n is an integer >=0 counting the filters (for
    identification purposes) and j is an integer >= 0 denoting the maximal
    subsampling 2**j which can be performed on a signal convolved with this
    filter without aliasing.

    Parameters
    ----------
    J : int
        maximal scale of the scattering (controls the number of wavelets)
    Q : int / tuple[int]
        The number of first-order wavelets per octave. Defaults to `1`.
        If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
        second-order wavelets per octave (which defaults to `1`).
            - Q1: For audio signals, a value of `>= 12` is recommended in
              order to separate partials.
            - Q2: Recommended `2` or `1` for most applications.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
    r_psi : float / tuple[float], optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets),
        and stability against time-warp deformations (larger r_psi improves it).
        Defaults to sqrt(0.5).
        Tuple sets separately for first- and second-order filters.
    sigma0 : float, optional
        frequential width of the low-pass filter at scale J=0
        (the subsequent widths are defined by sigma_J = sigma0 / 2^J).
        Defaults to 1e-1
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 4.
    xi_min : float, optional
        Lower bound on `xi` to ensure every bandpass is a valid wavelet
        (doesn't peak at FFT bin 1) within `2*len(x)` padding.

    Returns
    -------
    sigma_low : float
        frequential width of the low-pass filter
    xi1 : list[float]
        Center frequencies of the first order filters.
    sigma1 : list[float]
        Frequential widths of the first order filters.
    j1 : list[int]
        Subsampling factors of the first order filters.
    is_cqt1 : list[bool]
        Constant Q Transform construction flag of the first order filters.
    xi2, sigma2, j2, is_cqt2 :
        `xi1, sigma1, j1, is_cqt1` for second order filters.

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    filter_bank.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    Q1, Q2 = Q if isinstance(Q, tuple) else (Q, 1)
    J1, J2 = J if isinstance(J, tuple) else (J, J)
    r_psi1, r_psi2 = r_psi if isinstance(r_psi, tuple) else (r_psi, r_psi)
    if Q1 < 1 or Q2 < 1:
        raise ValueError('Q should always be >= 1, got {}'.format(Q))

    # lower bound of band-pass filter frequential widths:
    # for default T = 2**(J), this coincides with sigma_low
    sigma_min1 = sigma0 / math.pow(2, J1)
    sigma_min2 = sigma0 / math.pow(2, J2)

    xi1s, sigma1s, j1s, is_cqt1s = compute_params_filterbank(
        sigma_min1, Q1, r_psi=r_psi1, alpha=alpha, J_pad=J_pad)
    xi2s, sigma2s, j2s, is_cqt2s = compute_params_filterbank(
        sigma_min2, Q2, r_psi=r_psi2, alpha=alpha, J_pad=J_pad)

    # width of the low-pass filter
    sigma_low = sigma0 / T
    return sigma_low, xi1s, sigma1s, j1s, is_cqt1s, xi2s, sigma2s, j2s, is_cqt2s


def scattering_filter_factory(N, J_support, J_scattering, Q, T,
                              r_psi=math.sqrt(0.5), criterion_amplitude=1e-3,
                              normalize='l1', max_subsampling=None, sigma0=0.1,
                              alpha=4., P_max=5, eps=1e-7, **kwargs):
    """
    Builds in Fourier the Morlet filters used for the scattering transform.

    Each single filter is provided as a dictionary with the following keys:
    * 'xi': central frequency, defaults to 0 for low-pass filters.
    * 'sigma': frequential width
    * k where k is an integer bounded below by 0. The maximal value for k
        depends on the type of filter, it is dynamically chosen depending
        on max_subsampling and the characteristics of the filters.
        Each value for k is an array (or tensor) of size 2**(J_support - k)
        containing the Fourier transform of the filter after subsampling by
        2**k
    * 'width': temporal width (interval of temporal invariance, i.e. its "T")
    * 'support': temporal support (interval outside which wavelet is ~0)
    # TODO 'j'

    Parameters
    ----------
    J_support : int
        2**J_support is the desired support size of the filters
    J_scattering : int
        parameter for the scattering transform (2**J_scattering
        corresponds to maximal temporal support of any filter)
    Q : int >= 1 / tuple[int]
        The number of first-order wavelets per octave. Defaults to `1`.
        If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
        second-order wavelets per octave (which defaults to `1`).
            - Q1: For audio signals, a value of `>= 12` is recommended in
              order to separate partials.
            - Q2: Recommended `1` for most (`Scattering1D)` applications.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
    r_psi : float / tuple[float], optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets),
        and stability against time-warp deformations (larger r_psi improves it).
        Defaults to sqrt(0.5).
        Tuple sets separately for first- and second-order filters.
    criterion_amplitude : float, optional
        Represents the numerical error which is allowed to be lost after
        convolution and padding. Defaults to 1e-3.
    normalize : string, optional
        Normalization convention for the filters (in the
        temporal domain). Supported values include 'l1' and 'l2'; a ValueError
        is raised otherwise. Defaults to 'l1'.
    max_subsampling: int or None, optional
        maximal dyadic subsampling to compute, in order
        to save computation time if it is not required. Defaults to None, in
        which case this value is dynamically adjusted depending on the filters.
    sigma0 : float, optional
        parameter controlling the frequential width of the
        low-pass filter at J_scattering=0; at a an absolute J_scattering, it
        is equal to sigma0 / 2**J_scattering. Defaults to 1e-1
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 4.
    P_max : int, optional
        maximal number of periods to use to make sure that the Fourier
        transform of the filters is periodic. P_max = 5 is more than enough for
        double precision. Defaults to 5. Should be >= 1
    eps : float, optional
        required machine precision for the periodization (single
        floating point is enough for deep learning applications).
        Defaults to 1e-7

    Returns
    -------
    phi_f : dictionary
        a dictionary containing the low-pass filter at all possible
        subsamplings. See above for a description of the dictionary structure.
        The possible subsamplings are controlled by the inputs they can
        receive, which correspond to the subsamplings performed on top of the
        1st and 2nd order transforms.
    psi1_f : dictionary
        a dictionary containing the band-pass filters of the 1st order,
        only for the base resolution as no subsampling is used in the
        scattering tree.
        Each value corresponds to a dictionary for a single filter, see above
        for an exact description.
        The keys of this dictionary are of the type (j, n) where n is an
        integer counting the filters and j the maximal dyadic subsampling
        which can be performed on top of the filter without aliasing.
    psi2_f : dictionary
        a dictionary containing the band-pass filters of the 2nd order
        at all possible subsamplings. The subsamplings are determined by the
        input they can receive, which depends on the scattering tree.
        Each value corresponds to a dictionary for a single filter, see above
        for an exact description.
        The keys of this dictionary are of th etype (j, n) where n is an
        integer counting the filters and j is the maximal dyadic subsampling
        which can be performed on top of this filter without aliasing.

    References
    ----------
      1. Convolutional operators in the time-frequency domain, V. Lostanlen,
         PhD Thesis, 2017
         https://tel.archives-ouvertes.fr/tel-01559667
      2. This is a modification of
         https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
         filter_bank.py
         Kymatio, (C) 2018-present. The Kymatio developers.
    """
    # compute the spectral parameters of the filters
    (sigma_low, xi1s, sigma1s, j1s, is_cqt1s, xi2s, sigma2s, j2s, is_cqt2s
     ) = calibrate_scattering_filters(J_scattering, Q, T, r_psi=r_psi,
                                      sigma0=sigma0, alpha=alpha, J_pad=J_support)

    # instantiate the dictionaries which will contain the filters
    phi_f = {}
    psi1_f = []
    psi2_f = []

    # compute the band-pass filters of the second order,
    # which can take as input a subsampled
    N_pad = 2**J_support
    for (n2, j2) in enumerate(j2s):
        # compute the current value for the max_subsampling,
        # which depends on the input it can accept.
        if max_subsampling is None:
            possible_subsamplings_after_order1 = [
                j1 for j1 in j1s if j2 >= j1]
            if len(possible_subsamplings_after_order1) > 0:
                max_sub_psi2 = max(possible_subsamplings_after_order1)
            else:
                max_sub_psi2 = 0
        else:
            max_sub_psi2 = max_subsampling

        # We first compute the filter without subsampling
        psi_f = {}
        psi_f[0] = morlet_1d(
            N_pad, xi2s[n2], sigma2s[n2], normalize=normalize,
            P_max=P_max, eps=eps)
        # compute the filter after subsampling at all other subsamplings
        # which might be received by the network, based on this first filter
        for subsampling in range(1, max_sub_psi2 + 1):
            factor_subsampling = 2**subsampling
            psi_f[subsampling] = periodize_filter_fourier(
                psi_f[0], nperiods=factor_subsampling)
        psi2_f.append(psi_f)

    # for the 1st order filters, the input is not subsampled so we
    # can only compute them with N_pad=2**J_support
    for (n1, j1) in enumerate(j1s):
        psi1_f.append({0: morlet_1d(
            N_pad, xi1s[n1], sigma1s[n1], normalize=normalize,
            P_max=P_max, eps=eps)})

    # compute the low-pass filters phi
    # Determine the maximal subsampling for phi, which depends on the
    # input it can accept (both 1st and 2nd order)
    log2_T = math.floor(math.log2(T))
    if max_subsampling is None:
        max_subsampling_after_psi1 = max(j1s)
        max_subsampling_after_psi2 = max(j2s)
        max_sub_phi = min(max(max_subsampling_after_psi1,
                              max_subsampling_after_psi2), log2_T)
    else:
        max_sub_phi = max_subsampling

    # compute the filters at all possible subsamplings
    phi_f[0] = gauss_1d(N_pad, sigma_low, normalize=normalize, P_max=P_max, eps=eps)
    for subsampling in range(1, max_sub_phi + 1):
        factor_subsampling = 2**subsampling
        # compute the low_pass filter
        phi_f[subsampling] = periodize_filter_fourier(
            phi_f[0], nperiods=factor_subsampling)

    # Embed the meta information within the filters
    ca = dict(criterion_amplitude=criterion_amplitude)
    s0ca = dict(N=N, sigma0=sigma0, criterion_amplitude=criterion_amplitude)
    for (n1, j1) in enumerate(j1s):
        psi1_f[n1]['xi'] = xi1s[n1]
        psi1_f[n1]['sigma'] = sigma1s[n1]
        psi1_f[n1]['j'] = j1
        psi1_f[n1]['is_cqt'] = is_cqt1s[n1]
        psi1_f[n1]['width'] = {0: 2*compute_temporal_width(
            psi1_f[n1][0], **s0ca)}
        psi1_f[n1]['support'] = {0: 2*compute_temporal_support(
            psi1_f[n1][0], **ca)}

    for (n2, j2) in enumerate(j2s):
        psi2_f[n2]['xi'] = xi2s[n2]
        psi2_f[n2]['sigma'] = sigma2s[n2]
        psi2_f[n2]['j'] = j2
        psi2_f[n2]['is_cqt'] = is_cqt2s[n2]
        psi2_f[n2]['width'] = {}
        psi2_f[n2]['support'] = {}
        for k in psi2_f[n2]:
            if isinstance(k, int):
                psi2_f[n2]['width'][k] = 2*compute_temporal_width(
                    psi2_f[n2][k], **s0ca)
                psi2_f[n2]['support'][k] = 2*compute_temporal_support(
                    psi2_f[n2][k], **ca)

    phi_f['xi'] = 0.
    phi_f['sigma'] = sigma_low
    phi_f['j'] = log2_T
    phi_f['width'] = 2*compute_temporal_width(phi_f[0], **s0ca)
    phi_f['support'] = 2*compute_temporal_support(phi_f[0], **ca)

    # return results
    return phi_f, psi1_f, psi2_f


#### Energy renormalization ##################################################
def energy_norm_filterbank_tm(psi1_f, psi2_f, phi_f, J, log2_T):
    """Energy-renormalize temporal filterbank; used by `base_frontend`.
    See `help(wavespin.scattering1d.filter_bank.energy_norm_filterbank)`.
    """
    # in case of `trim_tm` for JTFS
    # phi = phi_f[0][0] if isinstance(phi_f[0], list) else phi_f[0]  # TODO
    phi = None
    kw = dict(phi_f=phi, log2_T=log2_T, passes=3)
    psi1_f0 = [p[0] for p in psi1_f]
    psi2_f0 = [p[0] for p in psi2_f]

    energy_norm_filterbank(psi1_f0, J=J[0], **kw)
    scaling_factors2 = energy_norm_filterbank(psi2_f0, J=J[1], **kw)

    # apply unsubsampled scaling factors on subsampled
    for n2 in range(len(psi2_f)):
        for k in psi2_f[n2]:
            if isinstance(k, int) and k != 0:
                psi2_f[n2][k] *= scaling_factors2[0][n2]


def energy_norm_filterbank_fr(psi1_f_fr_up, psi1_f_fr_dn, phi_f_fr,
                              J_fr, log2_F, sampling_psi_fr):
    """Energy-renormalize frequential filterbank; used by `base_frontend`.
    See `help(wavespin.scattering1d.filter_bank.energy_norm_filterbank)`.
    """
    psi_id_max = max(psi_id for psi_id in psi1_f_fr_up
                     if isinstance(psi_id, int))
    psi_id_break = None
    for psi_id in range(psi_id_max + 1):
        psi_fs_up = psi1_f_fr_up[psi_id]
        psi_fs_dn = psi1_f_fr_dn[psi_id]

        if len(psi_fs_up) <= 3:  # possible with `sampling_psi_fr = 'exclude'`
            if psi_id == 0:
                raise Exception("largest scale filterbank must have >=4 filters")
            psi_id_break = psi_id
            break
        phi_f = None  # not worth the hassle to account for
        passes = 10  # can afford to do more without performance hit
        is_recalibrate = bool(sampling_psi_fr == 'recalibrate')
        scaling_factors = energy_norm_filterbank(psi_fs_up, psi_fs_dn, phi_f,
                                                 J_fr, log2_F,
                                                 is_recalibrate=is_recalibrate,
                                                 passes=passes)

    # we stopped normalizing when there were <= 3 filters ('exclude'),
    # but must still normalize the rest, so reuse factors from when we last had >3
    if psi_id_break is not None:
        for psi_id in range(psi_id_break, psi_id_max + 1):
            if psi_id not in psi1_f_fr_dn:
                continue
            for n1_fr in range(len(psi1_f_fr_dn[psi_id])):
                psi1_f_fr_up[psi_id][n1_fr] *= scaling_factors[0][n1_fr]
                psi1_f_fr_dn[psi_id][n1_fr] *= scaling_factors[1][n1_fr]


def energy_norm_filterbank(psi_fs0, psi_fs1=None, phi_f=None, J=None, log2_T=None,
                           r_th=.3, is_recalibrate=False, passes=3,
                           scaling_factors=None):
    """Rescale wavelets such that their frequency-domain energy sum
    (Littlewood-Paley sum) peaks at 2 for an analytic-only filterbank
    (e.g. time scattering for real inputs) or 1 for analytic + anti-analytic.
    This makes the filterbank energy non-expansive.

    Parameters
    ----------
    psi_fs0 : list[np.ndarray]
        Analytic filters (also spin up for frequential).

    psi_fs1 : list[np.ndarray] / None
        Anti-analytic filters (spin down). If None, filterbank is treated as
        analytic-only, and LP peaks are scaled to 2 instead of 1.

    phi_f : np.ndarray / None
        Lowpass filter. If `log2_T < J`, will exclude from computation as
        it will excessively attenuate low frequency bandpasses.

    J, log2_T : int, int
        See `phi_f`. For JTFS frequential scattering these are `J_fr, log2_F`.

    r_th : float
        Redundancy threshold, determines whether "Nyquist correction" is done
        (see Algorithm below).

    passes : int
        Number of times to call this function recursively; see Algorithm.

    scaling_factors : None / dict[float]
        Used internally if `passes > 1`.

    Returns
    -------
    scaling_factors : None / dict[float]
        Used internally if `passes > 1`.

    Algorithm
    ---------
    Wavelets are scaled by maximum of *neighborhood* LP sum - precisely, LP sum
    spanning from previous to next peak location relative to wavelet of interest:
    `max(lp_sum[peak_idx[n + 1]:peak_idx[n - 1]])`. This reliably accounts for
    discretization artifacts, including the non-CQT portion.

    "Nyquist correction" is done for the highest frequency wavelet; since it
    has no "preceding" wavelet, it's its own right bound (analytic; left for
    anti-), which overestimates necessary rescaling and makes resulting LP sum
    peak above target for the *next* wavelet. This correction is done only if
    the filterbank is below a threshold redundancy (empirically determined
    `r_th=.3`), since otherwise the direct estimate is accurate.

    Multiple "passes" are done to improve overall accuracy, as not all edge
    case behavior is accounted for in one go (which is possible but complicated);
    the computational burden is minimal.
    """
    from ..toolkit import compute_lp_sum

    def norm_filter(psi_fs, peak_idxs, lp_sum, n, s_idx=0):
        # higher freq idx
        if n - 1 in peak_idxs:
            # midpoint
            pi0, pi1 = peak_idxs[n], peak_idxs[n - 1]
            if pi1 == pi0:
                # handle duplicate peaks
                lookback = 2
                while n - lookback in peak_idxs:
                    pi1 = peak_idxs[n - lookback]
                    if pi1 != pi0:
                        break
                    lookback += 1
            midpt = (pi0 + pi1) / 2
            # round closer to Nyquist
            a = (math.ceil(midpt) if s_idx == 0 else
                 math.floor(midpt))
        else:
            a = peak_idxs[n]

        # lower freq idx
        if n + 1 in peak_idxs:
            if n == 0 and nyquist_correction:
                b = a + 1 if s_idx == 1 else a - 1
            else:
                b = peak_idxs[n + 1]
        else:
            b = (None if s_idx == 1 else
                 1)  # exclude dc

        # peak duplicate
        if a == b:
            if s_idx == 1:
                b += 1
            else:
                b -= 1
        start, end = (a, b) if s_idx == 1 else (b, a)

        # include endpoint
        if end is not None:
            end += 1

        # if we're at endpoints, don't base estimate on single point
        if start is None:  # left endpoint
            end = max(end, 2)
        elif end is None:  # right endpoint
            start = min(start, len(lp_sum) - 1)
        elif end - start == 1:
            if start == 0:
                end += 1
            elif end == len(lp_sum) - 1:
                start -= 1

        lp_max = lp_sum[start:end].max()
        factor = np.sqrt(peak_target / lp_max)
        psi_fs[n] *= factor
        if n not in scaling_factors[s_idx]:
            scaling_factors[s_idx][n] = 1
        scaling_factors[s_idx][n] *= factor

    def correct_nyquist(psi_fs_all, peak_idxs, lp_sum):
        def _do_correction(start, end, s_idx=0):
            lp_max = lp_sum[start:end].max()
            factor = np.sqrt(peak_target / lp_max)
            for n in (0, 1):
                psi_fs[n] *= factor
                scaling_factors[s_idx][n] *= factor

        # first (Nyquist-nearest) psi rescaling may drive LP sum above bound
        # for second psi, since peak was taken only at itself
        if analytic_only:
            psi_fs = psi_fs_all
            # include endpoint
            start, end = peak_idxs[2], peak_idxs[0] + 1
            _do_correction(start, end)
        else:
            for s_idx, psi_fs in enumerate(psi_fs_all):
                a = peak_idxs[s_idx][0]
                b = peak_idxs[s_idx][2]
                start, end = (a, b) if s_idx == 1 else (b, a)
                # include endpoint
                end += 1
                _do_correction(start, end, s_idx)

    # run input checks #######################################################
    assert len(psi_fs0) >= 4, (
        "must have at least 4 filters in filterbank (got %s) " % len(psi_fs0)
        + "try increasing J or Q")
    if psi_fs1 is not None:
        assert len(psi_fs0) == len(psi_fs1), (
            "analytic & anti-analytic filterbanks "
            "must have same number of filters")
    # assume same overlap for analytic and anti-analytic
    r = compute_filter_redundancy(psi_fs0[0], psi_fs0[1])
    nyquist_correction = bool(r < r_th)

    # as opposed to `analytic_and_anti_analytic`
    analytic_only = bool(psi_fs1 is None)
    peak_target = 2 if analytic_only else 1

    # execute ################################################################
    # store rescaling factors
    if scaling_factors is None:  # else means passes>1
        scaling_factors = {0: {}, 1: {}}

    # compute peak indices
    peak_idxs = {}
    if analytic_only:
        psi_fs_all = psi_fs0
        for n, psi_f in enumerate(psi_fs0):
            peak_idxs[n] = np.argmax(psi_f)
    else:
        psi_fs_all = (psi_fs0, psi_fs1)
        for s_idx, psi_fs in enumerate(psi_fs_all):
            peak_idxs[s_idx] = {}
            for n, psi_f in enumerate(psi_fs):
                peak_idxs[s_idx][n] = np.argmax(psi_f)

    # warn if there are 3 or more shared peaks
    pidxs_either = list((peak_idxs if analytic_only else peak_idxs[0]).values())
    th = 3 if is_recalibrate else 2  # at least one overlap likely in recalibrate
    if any(pidxs_either.count(idx) >= th for idx in pidxs_either):
        pad_varname = "max_pad_factor" if analytic_only else "max_pad_factor_fr"
        warnings.warn(f"Found >={th} wavelets with same peak freq, most likely "
                      f"per too small `{pad_varname}`; energy norm may be poor")

    # ensure LP sum peaks at 2 (analytic-only) or 1 (analytic + anti-analytic)
    def get_lp_sum():
        if analytic_only:
            return compute_lp_sum(psi_fs0, phi_f, J, log2_T,
                                  fold_antianalytic=True)
        else:
            return (compute_lp_sum(psi_fs0, phi_f, J, log2_T) +
                    compute_lp_sum(psi_fs1))

    lp_sum = get_lp_sum()
    assert len(lp_sum) % 2 == 0, "expected even-length wavelets"
    if analytic_only:  # time scattering
        for n in range(len(psi_fs0)):
            norm_filter(psi_fs0, peak_idxs, lp_sum, n)
    else:  # frequential scattering
        for s_idx, psi_fs in enumerate(psi_fs_all):
            for n in range(len(psi_fs)):
                norm_filter(psi_fs, peak_idxs[s_idx], lp_sum, n, s_idx)

    if nyquist_correction:
        lp_sum = get_lp_sum()  # compute against latest
        correct_nyquist(psi_fs_all, peak_idxs, lp_sum)

    if passes == 1:
        return scaling_factors
    return energy_norm_filterbank(psi_fs0, psi_fs1, phi_f, J, log2_T,
                                  r_th, is_recalibrate, passes - 1,
                                  scaling_factors)


#### misc / long #############################################################
def compute_filter_redundancy(p0_f, p1_f):
    """Measures "redundancy" as overlap of energies. Namely, ratio of
    product of energies to mean of energies of Frequency-domain filters
    `p0_f` and `p1_f`.
    """
    p0sq, p1sq = np.abs(p0_f)**2, np.abs(p1_f)**2
    # energy overlap relative to sum of individual energies
    r = np.sum(p0sq * p1sq) / ((p0sq.sum() + p1sq.sum()) / 2)
    return r


def compute_temporal_width(p_f, N=None, pts_per_scale=6, fast=True,
                           sigma0=.1, criterion_amplitude=1e-3):
    """Measures "width" in terms of amount of invariance imposed via convolution.
    See below for detailed description.

    Parameters
    ----------
    p_f: np.ndarray
        Frequency-domain filter of length >= N. "Length" must be along dim0,
        i.e. `(freqs, ...)`.

    N: int / None
        Unpadded output length. (In scattering we convolve at e.g. x4 input's
        length, then unpad to match input's length).
        Defaults to `len(p_f) // 2`.

    pts_per_scale: int
        Used only in `fast=False`: number of Gaussians generated per dyadic
        scale. Greater improves approximation accuracy but is slower.

    sigma0, criterion_amplitude: float, float
        See `help(wavespin.scattering1d.filter_bank.gauss_1d)`. Parameters
        defining the Gaussian lowpass used as reference for computing `width`.
        That is, `width` is defined *in terms of* this Gaussian.

    Returns
    -------
    width: int
        The estimated width of `p_f`.

    Motivation
    ----------
    The measure is, a relative L2 distance (Euclidean distance relative to
    input norms) between inner products of `p_f` with an input, at different
    time shifts (i.e. `L2(A, B); A = sum(p(t) * x(t)), B = sum(p(t) * x(t - T))`).
      - The time shift is made to be "maximal", i.e. half of unpadded output
        length (`N/2`), which provides the most sensitive measure to "width".
      - The input is a Dirac delta, thus we're really measuring distances between
        impulse responses of `p_f`, or of `p_f` with itself. This provides a
        measure that's close to that of input being WGN, many-trials averaged.

    This yields `l2_reference`. It's then compared against `l2_Ts`, which is
    a list of measures obtained the same way except replacing `p_f` with a fully
    decayed (undistorted) Gaussian lowpass - and the `l2_T` which most closely
    matches `l2_reference` is taken to be the `width`.

    The interpretation is, "width" of `p_f` is the width of the (properly decayed)
    Gaussian lowpass that imposes the same amount of invariance that `p_f` does.

    Algorithm
    ---------
    The actual algorithm is different, since L2 is a problematic measure with
    unpadding; there's ambiguity: `T=1` can be measured as "closer" to `T=64`
    than `T=63`). Namely, we take inner product (similarity) of `p_f` with
    Gaussians at varied `T`, and the largest such product is the `width`.
    The result is very close to that described under "Motivation", but without
    the ambiguity that requires correction steps.

    Fast algorithm
    --------------
    Approximates `fast=False`. If `p_f` is fully decayed, the result is exactly
    same as `fast=False`. If `p_f` is very wide (nearly global average), the
    result is also same as `fast=False`. The disagreement is in the intermediate
    zone, but is not significant.

    We compute `ratio = p_t.max() / p_t.min()`, and compare against a fully
    decayed reference. For the intermediate zone, a quadratic interpolation
    is used to approximate `fast=False`.

    Assumption
    ----------
    `abs(p_t)`, where `p_t = ifft(p_f)`, is assumed to be Gaussian-like.
    An ideal measure is devoid of such an assumption, but is difficult to devise
    in finite sample settings.

    Note
    ----
    `width` saturates at `N` past a certain point in "incomplete decay" regime.
    The intution is, in incomplete decay, the measure of width is ambiguous:
    for one, we lack compact support. For other, the case `width == N` is a
    global averaging (simple mean, dc, etc), which is really `width = inf`.

    If we let the algorithm run with unrestricted `T_max`, we'll see `width`
    estimates blow up as `T -> N` - but we only seek to quantify invariance
    up to `N`. Also, Gaussians of widths `T = N - const` and `T = N` have
    very close L2 measures up to a generous `const`; see `test_global_averaging()`
    in `tests/scattering1d/test_jtfs.py` for `T = N - 1` (and try e.g. `N - 64`).
    """
    if len(p_f) == 1:  # edge case
        return 1

    # obtain temporal filter
    p_f = p_f.squeeze()
    p_t = np.abs(ifft(p_f))

    # relevant params
    Np = len(p_f)
    if N is None:
        N = Np // 2
    ca = dict(criterion_amplitude=criterion_amplitude)

    # compute "complete decay" factor
    uses_defaults = bool(sigma0 == .1 and criterion_amplitude == 1e-3)
    if uses_defaults:
        # precomputed
        complete_decay_factor = 16
        fast_approx_amp_ratio = 0.8208687174155399
    else:
        if fast:  # TODO unnecessary?
            raise ValueError("`fast` requires using default values of "
                             "`sigma0` and `criterion_amplitude`.")
        T = Np
        phi_f_fn = lambda Np_phi: gauss_1d(Np_phi, sigma0 / T)
        Np_min = compute_minimum_required_length(phi_f_fn, Np, **ca)
        complete_decay_factor = 2 ** math.ceil(math.log2(Np_min / Np))
        phi_t = phi_f_fn(Np_min)
        fast_approx_amp_ratio = phi_t[T] / phi_t[0]

    if fast:
        ratio = (p_t / p_t[0])[:len(p_t)//2]  # assume ~symmetry about halflength
        rmin = ratio.min()
        if rmin > fast_approx_amp_ratio:
            # equivalent of `not complete_decay`
            th_global_avg = .96
            # never sufficiently decays
            if rmin > th_global_avg:
                return N
            # quadratic interpolation
            # y0 = A + B*x0^2
            # y1 = A + B*x1^2
            # B = (y0 - y1) / (x0^2 - x1^2)
            # A = y0 - B*x0^2
            y0 = .5 * Np
            y1 = N
            x0 = fast_approx_amp_ratio
            x1 = th_global_avg
            B = (y0 - y1) / (x0**2 - x1**2)
            A = y0 - B*x0**2
            T_est = A + B*rmin**2
            # do not exceed `N`
            width = min(T_est, N)
        else:
            width = np.argmin(np.abs(ratio - fast_approx_amp_ratio))
        return width

    # if complete decay, search within length's scale
    support = 2 * compute_temporal_support(p_f, **ca)
    complete_decay = bool(support != Np)
    too_short = bool(N == 2 or Np == 2)
    if too_short:
        return (1 if complete_decay else 2)
    elif not complete_decay:
        # cannot exceed by definition
        T_max = N
        # if it were less, we'd have `complete_decay`
        T_min = 2 ** math.ceil(math.log2(Np / complete_decay_factor))
    else:  # complete decay
        # if it were more, we'd have `not complete_decay`
        # the `+ 1` is to be safe in edge cases
        T_max = 2 ** math.ceil(math.log2(Np / complete_decay_factor) + 1)
        # follows from relation of `complete_decay_factor` to `width`
        # `width \propto support`, `support = complete_decay_factor * stuff`
        # (asm. full decay); so approx `support ~= complete_decay_factor * width`
        T_min = 2 ** math.floor(math.log2(support / complete_decay_factor))
    T_min = max(min(T_min, T_max // 2), 1)  # ensure max > min and T_min >= 1
    T_max = max(T_max, 2)  # ensure T_max >= 2
    T_min_orig, T_max_orig = T_min, T_max

    n_scales = math.log2(T_max) - math.log2(T_min)
    search_pts = int(n_scales * pts_per_scale)

    # search T ###############################################################
    def search_T(T_min, T_max, search_pts, log):
        Ts = (np.linspace(T_min, T_max, search_pts) if not log else
              np.logspace(np.log10(T_min), np.log10(T_max), search_pts))
        Ts = np.unique(np.round(Ts).astype(int))

        Ts_done = []
        corrs = []
        for T_test in Ts:
            N_phi = max(int(T_test * complete_decay_factor), Np)
            phi_f = gauss_1d(N_phi, sigma=sigma0 / T_test)
            phi_t = ifft(phi_f).real

            trim = min(min(len(p_t), len(phi_t))//2, N)
            p0, p1 = p_t[:trim], phi_t[:trim]
            p0 /= np.linalg.norm(p0)  # /= sqrt(sum(x**2))
            p1 /= np.linalg.norm(p1)
            corrs.append((p0 * p1).sum())
            Ts_done.append(T_test)

        T_est = int(round(Ts_done[np.argmax(corrs)]))
        T_stride = int(Ts[1] - Ts[0])
        return T_est, T_stride

    # first search in log space
    T_est, _ = search_T(T_min, T_max, search_pts, log=True)
    # refine search, now in linear space
    T_min = max(2**math.floor(math.log2(max(T_est - 1, 1))), T_min_orig)
    # +1 to ensure T_min != T_max
    T_max = min(2**math.ceil(math.log2(T_est + 1)), T_max_orig)
    # only one scale now
    search_pts = pts_per_scale
    T_est, T_stride = search_T(T_min, T_max, search_pts, log=False)
    # only within one zoom
    diff = pts_per_scale // 2
    T_min, T_max = max(T_est - diff, 1), max(T_est + diff - 1, 3)
    T_est, _ = search_T(T_min, T_max, search_pts, log=False)

    width = T_est
    return width
