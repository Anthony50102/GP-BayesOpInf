# step1_5_estimate_priors.py
import numpy as np
import matplotlib.pyplot as plt
import mpmath  
import sys
from gpytorch.priors import NormalPrior
import torch

pi = np.pi

def bgls(t, y, err, plow=0.5, phigh=100, ofac=1):
    """
    Compute the Bayesian Generalised Lomb-Scargle periodogram.
    
    Parameters:
      t     : array_like, times of observations
      y     : array_like, observed values
      err   : array_like, errors on observations (assumed to be constant if desired)
      plow  : float, lower bound of period (default 0.5)
      phigh : float, upper bound of period (default 100)
      ofac  : oversampling factor (default 1)
      
    Returns:
      periods : numpy array of periods (1/frequency)
      p       : normalized periodogram power corresponding to each period
    """
    n_steps = int(ofac * len(t) * (1.0/plow - 1.0/phigh))
    # Frequencies range from 1/phigh to 1/plow
    f = np.linspace(1.0/phigh, 1.0/plow, n_steps)
    omegas = 2.0 * pi * f

    err2 = err**2
    w = 1.0 / err2
    W = np.sum(w)
    bigY = np.sum(w * y)
    
    constants = []
    exponents = []
    
    for omega in omegas:
        theta = 0.5 * np.arctan2(np.sum(w * np.sin(2.0 * omega * t)),
                                  np.sum(w * np.cos(2.0 * omega * t)))
        x = omega * t - theta
        cosx = np.cos(x)
        sinx = np.sin(x)
        wcosx = w * cosx
        wsinx = w * sinx
        
        C = np.sum(wcosx)
        S = np.sum(wsinx)
        YCh = np.sum(y * wcosx)
        YSh = np.sum(y * wsinx)
        CCh = np.sum(wcosx * cosx)
        SSh = np.sum(wsinx * sinx)
        
        if CCh != 0 and SSh != 0:
            K = (C * C * SSh + S * S * CCh - W * CCh * SSh) / (2.0 * CCh * SSh)
            L = (bigY * CCh * SSh - C * YCh * SSh - S * YSh * CCh) / (CCh * SSh)
            M = (YCh * YCh * SSh + YSh * YSh * CCh) / (2.0 * CCh * SSh)
            constants.append(1.0 / np.sqrt(CCh * SSh * abs(K)))
        elif CCh == 0:
            K = (S * S - W * SSh) / (2.0 * SSh)
            L = (bigY * SSh - S * YSh) / SSh
            M = (YSh * YSh) / (2.0 * SSh)
            constants.append(1.0 / np.sqrt(SSh * abs(K)))
        elif SSh == 0:
            K = (C * C - W * CCh) / (2.0 * CCh)
            L = (bigY * CCh - C * YCh) / CCh
            M = (YCh * YCh) / (2.0 * CCh)
            constants.append(1.0 / np.sqrt(CCh * abs(K)))
        
        if K > 0:
            raise RuntimeError('K is positive. This should not happen.')
            
        exponents.append(M - L * L / (4.0 * K))
    
    constants = np.array(constants)
    exponents = np.array(exponents)

    # After calculating constants and exponents
    logp = np.log10(constants) + (exponents * np.log10(np.e))
    
    # Find the maximum logp value for normalization
    max_logp = np.max(logp[np.isfinite(logp)]) if np.any(np.isfinite(logp)) else 0.0
    
    # Normalize in log space by subtracting max_logp
    normalized_logp = logp - max_logp
    
    # Only convert to linear space at the end, with values now safely between 0 and 1
    p = 10.0 ** normalized_logp
    
    # Replace any remaining non-finite values with zeros
    p[~np.isfinite(p)] = 0.0
    
    periods = 1.0 / f
    return periods, p
    
    # logp = np.log10(constants) + (exponents * np.log10(np.e))
    
    # # Clip extremely low values to prevent underflow
    # logp = np.clip(logp, -300, 300)  # Reasonable range for float64
    
    # # Use exp + log approach instead of direct power for better numerical stability
    # p = 10.0 ** logp
    
    # # Check for NaN or inf values
    # mask = np.isfinite(p)
    # if not np.any(mask):
    #     print("Warning: All periodogram values are non-finite!")
    #     return periods, np.zeros_like(periods)
    
    # # Normalize only using finite values
    # max_p = np.max(p[mask]) if np.any(mask) else 1.0
    # if max_p > 0:
    #     p = p / max_p  # normalize the periodogram power
    # else:
    #     p = np.zeros_like(p)
        
    # # Replace any remaining non-finite values with zeros
    # p[~np.isfinite(p)] = 0.0
    
    # periods = 1.0 / f
    # return periods, p

def estimate_normal_period(periods, p):
    """
    Estimate the period (as a normal distribution) by computing the weighted mean 
    and standard deviation of the periodogram.
    
    Parameters:
      periods : numpy array of periods
      p       : numpy array of periodogram probabilities
      
    Returns:
      mean : weighted mean period
      std  : weighted standard deviation of the period
    """
    # Filter out any NaN values
    mask = np.isfinite(p) & np.isfinite(periods) & (p > 0)
    
    if not np.any(mask):
        print("Warning: No valid points for period estimation!")
        return float('nan'), float('nan')
    
    p_filtered = p[mask]
    periods_filtered = periods[mask]
    
    # Normalize probabilities for weighting
    p_sum = np.sum(p_filtered)
    if p_sum <= 0:
        return float('nan'), float('nan')
        
    p_norm = p_filtered / p_sum
    
    mean = np.sum(periods_filtered * p_norm)
    variance = np.sum(p_norm * (periods_filtered - mean)**2)
    std = np.sqrt(variance) if variance > 0 else 0.0
    
    return mean, std

def normal_pdf(x, mean, std):
    """
    Normal (Gaussian) probability density function.
    """
    return 1.0/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean)/std)**2)

def estimate_period(t, y, err=None, plow=1, phigh=100, ofac=10):
    priors = []
    
    # Iterate over each signal in y
    for i in range(len(y)):
        # Create error array for this specific signal
        current_err = np.full_like(t[i], 1) if err is None else err[i]
        
        # Pass the specific signal and its matching error array
        periods, p = bgls(t[i], y[i], current_err, plow=plow, phigh=phigh, ofac=ofac)
        period_mean, period_std = estimate_normal_period(periods, p)
        print(f"Estimated signal period to be {period_mean} with stddev of {period_std} for variable at index {i}")
        
        priors.append((period_mean/2, period_std))
        
    return priors

import numpy as np
import matplotlib.pyplot as plt
import mpmath  
import sys
from gpytorch.priors import NormalPrior
import torch

pi = np.pi

def bgls_diagnostic(t, y, err, plow=0.5, phigh=100, ofac=1):
    """
    Diagnostic version of BGLS that logs numerical issues
    """
    print("=== BGLS DIAGNOSTIC REPORT ===")
    print(f"Input shapes - t: {t.shape}, y: {y.shape}, err: {err.shape}")
    print(f"Period range: {plow} to {phigh}, ofac: {ofac}")
    
    # Compute values for numerical stability check
    min_t = np.min(t)
    max_t = np.max(t)
    range_t = max_t - min_t
    print(f"Time range: {min_t} to {max_t} (span: {range_t})")
    
    min_y = np.min(y)
    max_y = np.max(y)
    range_y = max_y - min_y
    print(f"Y range: {min_y} to {max_y} (span: {range_y})")
    
    n_steps = int(ofac * len(t) * (1.0/plow - 1.0/phigh))
    print(f"Computing {n_steps} frequency steps")
    
    # Frequencies range from 1/phigh to 1/plow
    f = np.linspace(1.0/phigh, 1.0/plow, n_steps)
    omegas = 2.0 * pi * f
    min_omega = np.min(omegas)
    max_omega = np.max(omegas)
    print(f"Omega range: {min_omega} to {max_omega}")
    
    # Check if omega * t will cause issues
    max_omega_t = max_omega * max_t
    print(f"Max omega*t: {max_omega_t} (potential for large phase values)")
    
    err2 = err**2
    w = 1.0 / err2
    W = np.sum(w)
    bigY = np.sum(w * y)
    print(f"W: {W}, bigY: {bigY}")
    
    constants = []
    exponents = []
    k_values = []
    l_values = []
    m_values = []
    theta_values = []
    cch_values = []
    ssh_values = []
    
    problem_frequencies = []
    
    for i, omega in enumerate(omegas):
        if i % 100 == 0:  # Print progress every 100 steps
            print(f"Processing frequency {i}/{len(omegas)}")
            
        # Phase calculation
        sin_term = np.sum(w * np.sin(2.0 * omega * t))
        cos_term = np.sum(w * np.cos(2.0 * omega * t))
        
        if i % 100 == 0:
            print(f"  Frequency {i}: sin_term={sin_term:.3e}, cos_term={cos_term:.3e}")
            
        theta = 0.5 * np.arctan2(sin_term, cos_term)
        theta_values.append(theta)
        
        x = omega * t - theta
        cosx = np.cos(x)
        sinx = np.sin(x)
        wcosx = w * cosx
        wsinx = w * sinx
        
        C = np.sum(wcosx)
        S = np.sum(wsinx)
        YCh = np.sum(y * wcosx)
        YSh = np.sum(y * wsinx)
        CCh = np.sum(wcosx * cosx)
        SSh = np.sum(wsinx * sinx)
        
        if i % 100 == 0:
            print(f"  C={C:.3e}, S={S:.3e}, YCh={YCh:.3e}, YSh={YSh:.3e}, CCh={CCh:.3e}, SSh={SSh:.3e}")
        
        cch_values.append(CCh)
        ssh_values.append(SSh)
        
        # Track very small values that might cause division issues
        if abs(CCh) < 1e-10 or abs(SSh) < 1e-10:
            problem_frequencies.append((i, omega, CCh, SSh))
            
        try:
            if CCh != 0 and SSh != 0:
                K = (C * C * SSh + S * S * CCh - W * CCh * SSh) / (2.0 * CCh * SSh)
                L = (bigY * CCh * SSh - C * YCh * SSh - S * YSh * CCh) / (CCh * SSh)
                M = (YCh * YCh * SSh + YSh * YSh * CCh) / (2.0 * CCh * SSh)
                constants.append(1.0 / np.sqrt(CCh * SSh * abs(K)))
            elif CCh == 0:
                K = (S * S - W * SSh) / (2.0 * SSh)
                L = (bigY * SSh - S * YSh) / SSh
                M = (YSh * YSh) / (2.0 * SSh)
                constants.append(1.0 / np.sqrt(SSh * abs(K)))
            elif SSh == 0:
                K = (C * C - W * CCh) / (2.0 * CCh)
                L = (bigY * CCh - C * YCh) / CCh
                M = (YCh * YCh) / (2.0 * CCh)
                constants.append(1.0 / np.sqrt(CCh * abs(K)))
            
            if K > 0:
                print(f"WARNING: K is positive at index {i}, omega={omega}: K={K}")
                
            k_values.append(K)
            l_values.append(L)
            m_values.append(M)
            
            # Calculate exponent term directly to check for potential overflow
            exponent = M - L * L / (4.0 * K)
            
            if not np.isfinite(exponent):
                print(f"WARNING: Non-finite exponent at index {i}, omega={omega}: M={M}, L={L}, K={K}, exponent={exponent}")
                
            exponents.append(exponent)
            
            if i % 100 == 0:
                print(f"  K={K:.3e}, L={L:.3e}, M={M:.3e}, exponent={exponent:.3e}")
                
        except Exception as e:
            print(f"ERROR at index {i}, omega={omega}: {str(e)}")
            print(f"  C={C:.3e}, S={S:.3e}, YCh={YCh:.3e}, YSh={YSh:.3e}")
            print(f"  CCh={CCh:.3e}, SSh={SSh:.3e}")
            # Add dummy values to maintain array length
            constants.append(0.0)
            exponents.append(0.0)
            k_values.append(float('nan'))
            l_values.append(float('nan'))
            m_values.append(float('nan'))
    
    # Convert to numpy arrays
    constants = np.array(constants)
    exponents = np.array(exponents)
    k_values = np.array(k_values)
    l_values = np.array(l_values)
    m_values = np.array(m_values)
    theta_values = np.array(theta_values)
    cch_values = np.array(cch_values)
    ssh_values = np.array(ssh_values)
    
    # Report on array statistics
    print("\n=== Array Statistics ===")
    
    def report_array(name, arr):
        if len(arr) == 0:
            print(f"{name}: Empty array")
            return
            
        finite_mask = np.isfinite(arr)
        finite_count = np.sum(finite_mask)
        
        if finite_count == 0:
            print(f"{name}: No finite values!")
            return
            
        finite_arr = arr[finite_mask]
        print(f"{name}: min={np.min(finite_arr):.3e}, max={np.max(finite_arr):.3e}, " +
              f"mean={np.mean(finite_arr):.3e}, median={np.median(finite_arr):.3e}")
        print(f"      non-finite: {len(arr) - finite_count}/{len(arr)}")
    
    report_array("constants", constants)
    report_array("exponents", exponents)
    report_array("K values", k_values)
    report_array("L values", l_values)
    report_array("M values", m_values)
    report_array("CCh values", cch_values)
    report_array("SSh values", ssh_values)
    
    # Check if any K values are very close to zero
    small_k_mask = np.abs(k_values) < 1e-10
    if np.any(small_k_mask):
        small_k_count = np.sum(small_k_mask)
        print(f"\nWARNING: {small_k_count} K values are very close to zero (< 1e-10)")
        print(f"This can cause overflow in the L*L/(4.0*K) term")
        
        # Find the 5 smallest abs(K) values
        abs_k = np.abs(k_values)
        smallest_indices = np.argsort(abs_k)[:5]
        print("5 smallest |K| values:")
        for idx in smallest_indices:
            if np.isfinite(k_values[idx]):
                omega = omegas[idx]
                period = 1.0/f[idx]
                print(f"  index {idx}: K={k_values[idx]:.3e}, L={l_values[idx]:.3e}, at period={period:.3f}")
    
    # Calculate logp with diagnostics
    print("\n=== LogP Calculation ===")
    log_constants = np.log10(constants)
    log_e_term = exponents * np.log10(np.e)
    
    report_array("log10(constants)", log_constants)
    report_array("exponents * log10(e)", log_e_term)
    
    logp = log_constants + log_e_term
    report_array("logp (before clipping)", logp)
    
    # Check for extreme values that would cause overflow
    extreme_mask = np.abs(logp) > 300
    if np.any(extreme_mask):
        extreme_count = np.sum(extreme_mask)
        print(f"\nWARNING: {extreme_count} logp values are extreme (|logp| > 300)")
        print("This will cause overflow/underflow when computing 10^logp")
        
        # Find the 5 most extreme logp values
        abs_logp = np.abs(logp)
        extreme_indices = np.argsort(-abs_logp)[:5]  # Minus sign to sort in descending order
        print("5 most extreme logp values:")
        for idx in extreme_indices:
            if np.isfinite(logp[idx]):
                omega = omegas[idx]
                period = 1.0/f[idx]
                print(f"  index {idx}: logp={logp[idx]:.3e}, at period={period:.3f}")
                print(f"    K={k_values[idx]:.3e}, L={l_values[idx]:.3e}, M={m_values[idx]:.3e}")
                print(f"    exponent={exponents[idx]:.3e}")
                print(f"    CCh={cch_values[idx]:.3e}, SSh={ssh_values[idx]:.3e}")
    
    # Now clip logp to prevent overflow
    logp = np.clip(logp, -300, 300)
    report_array("logp (after clipping)", logp)
    
    # Calculate p = 10^logp
    p = 10.0 ** logp
    report_array("p = 10^logp", p)
    
    # Check for NaN or inf in p
    mask = np.isfinite(p)
    if not np.any(mask):
        print("\nERROR: All periodogram values are non-finite!")
        return
    
    # Normalize p
    max_p = np.max(p[mask]) if np.any(mask) else 1.0
    if max_p > 0:
        p = p / max_p
    else:
        p = np.zeros_like(p)
    
    # Replace any remaining non-finite values with zeros
    p[~np.isfinite(p)] = 0.0
    report_array("p (normalized)", p)
    
    # Final result
    periods = 1.0 / f
    
    # Plot the periodogram to visualize the results
    plt.figure(figsize=(10, 6))
    plt.semilogx(periods, p)
    plt.xlabel('Period')
    plt.ylabel('Normalized Power')
    plt.title('BGLS Periodogram')
    plt.grid(True)
    plt.savefig('bgls_periodogram.png')
    print("\nPeriodogram plot saved as 'bgls_periodogram.png'")
    
    # Plot K values to identify issues
    plt.figure(figsize=(10, 6))
    finite_k = np.isfinite(k_values)
    plt.plot(periods[finite_k], k_values[finite_k])
    plt.xlabel('Period')
    plt.ylabel('K Value')
    plt.title('K Values vs Period')
    plt.grid(True)
    plt.savefig('bgls_k_values.png')
    print("K values plot saved as 'bgls_k_values.png'")
    
    # Plot exponents to identify issues
    plt.figure(figsize=(10, 6))
    finite_exp = np.isfinite(exponents)
    plt.plot(periods[finite_exp], exponents[finite_exp])
    plt.xlabel('Period')
    plt.ylabel('Exponent Value')
    plt.title('Exponents vs Period')
    plt.grid(True)
    plt.savefig('bgls_exponents.png')
    print("Exponents plot saved as 'bgls_exponents.png'")
    
    return periods, p

def estimate_period_diagnostic(t, y, err=None, plow=1, phigh=100, ofac=10):
    """
    Diagnostic version of estimate_period that provides detailed information
    about what might be causing numerical issues.
    """
    priors = []
    
    # Iterate over each signal in y
    for i in range(len(y)):
        print(f"\n=== Processing Signal {i+1}/{len(y)} ===")
        print(f"Signal shape: t[{i}]={t[i].shape}, y[{i}]={y[i].shape}")
        
        # Plot the input signal
        plt.figure(figsize=(10, 6))
        plt.plot(t[i], y[i])
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.title(f'Input Signal {i+1}')
        plt.grid(True)
        plt.savefig(f'signal_{i+1}.png')
        print(f"Input signal plot saved as 'signal_{i+1}.png'")
        
        # Create default error values if not provided
        current_err = np.full_like(t[i], 0.1) if err is None else err
        
        # Run diagnostic BGLS
        periods, p = bgls_diagnostic(t[i], y[i], current_err, plow=plow, phigh=phigh, ofac=ofac)
        
        # Estimate period using the improved function
        period_mean, period_std = estimate_normal_period(periods, p)
        print(f"Estimated signal period: mean={period_mean}, stddev={period_std}")
        
        if np.isfinite(period_mean) and np.isfinite(period_std) and period_std > 0:
            # Create a NormalPrior for the current signal and append it to the list
            priors.append(NormalPrior(torch.tensor(period_mean / 2), torch.tensor(period_std)))
            print(f"Created NormalPrior with mean={period_mean/2}, std={period_std}")
        else:
            print("WARNING: Could not create NormalPrior due to non-finite period estimation")
        
    return priors