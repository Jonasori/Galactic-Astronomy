"""
Basic MCMC pt. 2: Fit the fake data with a homemade MCMC routine

Develop a simple MCMC code to fit to data generated in utils.py
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import parabola, generate_fake_data, chi2
from utils import truncated_random_normal


# Calculate Chi-Squared
def chi2(data, model, sigma):
    # Takes two lists.
    if len(data) != len(model):
        return "Bad! No Good!"

    c = 0
    for i in range(len(data)):
        c += (data[i]-model[i])**2 * sigma**(-2)
        # print data[i], model[i], '\t', (data[i]-model[i])**2/sigma
    return c




# Choose a starting point
true_a = 1
true_b = 50.1
true_sigma = 10
# Generate some data
xs = np.arange(-20, 20)
ys = generate_fake_data(xs, true_a, true_b, true_sigma)

# Specify priors
priors_a = [-100, 100]
priors_b = [-100, 100]


def mcmc(xs, ys, priors_a, priors_b, sigma_a, sigma_b, sigma_data, nsteps=10000):
    """
    Run a Markov chain Monte Carlo.

    Args:
        xs, ys: input data (lists/numpy arrays)
        priors_a, priors_b: bounds on where the walkers can go
        sigma_a, sigma_b: FWHM for the Gaussian determining the next step
        sigma_data: noise level in the data
        nsteps (int): how many steps the walkers should take
    """


    # Give a starting point and calculate that initial chi2
    initial_a = np.random.uniform(priors_a[0], priors_a[1])
    initial_b = np.random.uniform(priors_b[0], priors_b[1])

    # Fill the first element of the position and chi2 arrays:
    a_vals, b_vals = [initial_a], [initial_b]

    first_model = parabola(xs, initial_a, initial_b)
    # This should not be a list. Maybe this is right.
    #chisqs = np.array(chi2(ys, first_model, sigma_data))
    chisqs = [chi2(ys, first_model, sigma_data)]

    # Initialize the acceptance/rejection counters:
    total_accepted = {'a': 0, 'b': 0}
    total_rejected = {'a': 0, 'b': 0}

    # Start the loop!
    i = 1
    while i < nsteps:
        print "Step: ", i
        # Propose a new step
        param_to_be_varied = np.random.choice(['a', 'b'], 1)[0]
        if param_to_be_varied == 'a':
            # a_new = np.random.normal(loc=a_vals[i-1], scale=sigma)
            a_new = truncated_random_normal(a_vals[-1],
                                            sigma_a, priors_a[0],
                                            priors_a[1])[0]

            new_step = np.array([a_new, b_vals[-1]])
        else:
            # b_new = np.random.normal(loc=b_vals[i-1], scale=sigma)
            b_new = truncated_random_normal(b_vals[-1],
                                            sigma_b, priors_b[0],
                                            priors_b[1])[0]

            new_step = np.array([a_vals[-1], b_new])

        # Calculate Chi-Squared for that new step
        model_new = parabola(xs, new_step[0], new_step[1])
        chisq_new = chi2(ys, model_new, sigma_data)

        # If the new one is an improvement, take it.
        if chisq_new < chisqs[-1]:
            a_vals.append(new_step[0])
            b_vals.append(new_step[1])
            chisqs.append(chisq_new)
            total_accepted[param_to_be_varied] += 1.

        # Otherwise, generate a random number in [0,1]
        else:
            r_num = np.random.random()
            # delta_chisq = chisq_old - chisq_new from eqn(13) of Ford 2005
            delta_chisq = (chisqs[-1] - chisq_new)
            alpha = np.exp(delta_chisq/2)

            # If alpha < random, reject this step.
            # It feels weird that we reject if alpha is smaller.
            if alpha < r_num:
                a_vals.append(a_vals[-1])
                b_vals.append(b_vals[-1])
                chisqs.append(chisqs[-1])
                total_rejected[param_to_be_varied] += 1.

            # If alpha > random, accept the new step.
            else:
                a_vals.append(new_step[0])
                b_vals.append(new_step[1])
                chisqs.append(chisq_new)
                total_accepted[param_to_be_varied] += 1.

        # Bump the counter
        i += 1

    # Final outputs
    acceptance_fraction_a = total_accepted['a']/(total_accepted['a'] + total_rejected['a'])
    acceptance_fraction_b = total_accepted['b']/(total_accepted['b'] + total_rejected['b'])
    print "\n\nFinal acceptance fraction A: ", acceptance_fraction_a
    print "\n\nFinal acceptance fraction B: ", acceptance_fraction_b

    total_acceptance_num = total_accepted['a'] + total_accepted['b']
    total_acceptance_den = total_accepted['a'] + total_accepted['b'] + total_rejected['a'] + total_rejected['b']
    total_acceptance = total_acceptance_num/total_acceptance_den
    print "\n\nTotal acceptance fraction (combined): ", total_acceptance

    final_output = {'a_vals_visited': a_vals,
                    'b_vals_visited': b_vals,
                    'chi2_vals': chisqs,
                    'acceptance_fraction_a': acceptance_fraction_b,
                    'acceptance_fraction_b': acceptance_fraction_b,
                    'acceptance_fraction_total': total_acceptance,
                    'nsteps': nsteps
                    }

    return final_output


# A generic plotter
def plot_whatever(xs, ys):
    plt.plot(xs, ys, '-og')
    plt.show(block=False)


# Plot it in parameter space
def plot_param_walk(run_output):
    nsteps = run_output['nsteps']
    burn_in_len = int(0.2 * nsteps)
    burn_in_len = 5000
    print burn_in_len
    a_vals = run_output['a_vals_visited'][burn_in_len:]
    b_vals = run_output['b_vals_visited'][burn_in_len:]
    plt.plot(a_vals, b_vals, '.k', alpha=0.01)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.show(block=False)


# Plot the resulting parabola
def plot_best_fit_model(run_output):
    a_vals = run_output['a_vals_visited']
    b_vals = run_output['b_vals_visited']
    chisqs = run_output['chi2_vals']

    min_vals = np.where(run_output['chi2_vals'] == np.min(run_output['chi2_vals']))

    #model_ys = parabola(xs, a_vals[min_vals[0,0]], b_vals[min_vals[0,1]])
    model_ys = parabola(xs, a_vals[min_vals[0][0]], b_vals[min_vals[0][1]])
    plt.plot(xs, model_ys, '-og')
    plt.plot(xs, ys, '-or')
    plt.show(block=False)


















# The End
