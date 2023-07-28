# import necessary libraries
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from numpy.polynomial import Polynomial



# Class for up-and-out American-style Asian puts option
class LSMC:
    def __init__(
        self,
        type,
        spot,
        strike,
        time,
        interest,
        dividend,
        volatility,
        period,
        simulations,

        degree=2,
    ):
        self.type = type
        self.spot_price = spot
        self.strike_price = strike
        self.time = time
        self.interest_rate = interest
        self.dividend_rate = dividend
        self.volatility_rate = volatility
        self.period = int(period)
        self.path = int(simulations)
        self.deltaT = time / period # In our case with an yearly exercise date we assume this value to be 1 therefore time interval equals the period
        self.discount = -self.interest_rate * self.deltaT
        self.degree = degree

    # Generate random path
    def random_path(self):

        # Randomized the path of Brownian Motions
        self.randomWalk = np.zeros((self.path, self.period + 1))

        # Initialize price
        self.randomWalk[:, 0] = self.spot_price

        # Random Walk each period. The price follow brownian motion
        for i in range(1, self.period + 1):
            self.randomWalk[:, i] = self.randomWalk[:, i - 1] * np.exp(
                (
                    self.interest_rate
                    - self.dividend_rate
                    - (self.volatility_rate ** 2) / 2
                )
                * self.deltaT
                + self.volatility_rate
                * np.sqrt(self.deltaT)
                * np.random.normal(size=self.path)
            )

    # Apply least square method on all path
    def pricing(self):

        # Payoff for each period of each paths
        if type == "p":
            self.payoff = np.maximum(0, self.strike_price - self.randomWalk)
        else:
            self.payoff = np.maximum(0, self.randomWalk - self.strike_price) # For Technologyoption the strike price might rever to todays value und self.randWalk to possible NPVs with invest times in the future

        Y = self.payoff[:,-1] 

        self.cash_flow_matrix = np.zeros_like(self.payoff)

        for i in range(self.period - 2, -1, -1): # before it was self.period - 2 why? -> numpy index is excluding thus i=49 is the 50fth value, for regression we start with the 49fht vlaue that corresponds to index i=48 (or i=N_periods-2)
            
            # Find valid path where the are positive cashflow thus the option is in the money
            hold_mask = self.payoff[:, i] > 0

            # Update current cashflows by discounted values from previous time step
            Y = Y * np.exp(self.discount)
            
            if np.count_nonzero(hold_mask) > self.degree: # keep sure that there is sufficent amount of point for the regression
                #Y[hold] are the random price path from random walk where the the option is in the money
            
                # Apply Least square method
                regression = Polynomial.fit(self.randomWalk[hold_mask, i], Y[hold_mask], self.degree)
                CV = regression(self.randomWalk[hold_mask, i])

                # Whether to exercise now
                Y[hold_mask] = np.where(
                    self.payoff[hold_mask, i] > CV, self.payoff[hold_mask, i], Y[hold_mask]
                )

                #Update cash flow matrix
                self.cash_flow_matrix[hold_mask,i] = Y[hold_mask]

        #Discount cash flows of each price path to time zero
        num_paths, num_time_steps = self.cash_flow_matrix.shape
        discount_factors = np.array([1/(1 + self.interest_rate) ** (t) for t in range(num_time_steps)])
        discounted_cash_flows = self.cash_flow_matrix * discount_factors

        # Sum the discounted cash flows along the time dimension to get the value at time zero
        self.cash_flow_matrix[:,0] =  np.sum(discounted_cash_flows, axis=1)

        # Calculate mean and variance of the stimulation
        self.Price = np.mean(self.cash_flow_matrix[:,0])
        self.std = np.std(self.cash_flow_matrix[:,0]) / np.sqrt(self.path)
        self.rmsre = np.sqrt(np.mean(((self.cash_flow_matrix[:,0] - self.Price) / self.Price) ** 2))



    # Plot the distibution
    def plot(self, bin=50, sampling=50):

        # draw histograms
        f1 = plt.figure(1)
        n, x, _ = plt.hist(self.cash_flow_matrix[:,0], bins=bin)

        x_bar = 0.5 * (x[1:] + x[:-1])

        plt.title("Distribution of Cash Flows")
        plt.xlabel("Discounted Cash Flow")
        plt.ylabel("Occurance")

        plt.figtext(
            0.01,
            0.01,
            f"estimated price: {round(self.Price, 4)}\nstandard deviation: {round(self.std, 4)}",
            fontsize=8,
        )

        # draw the fitting curve
        plt.plot(x_bar, n)

        # draw sampled path
        f2 = plt.figure(2)

        x = np.arange(0, self.period)

        for path in random.sample(range(self.path), sampling):
            plt.plot(x, self.randomWalk[path, :-1])

        plt.title("Sampled Path")
        plt.xlabel("Period")
        plt.ylabel("Price")

        plt.show()



### Utility functions that might be outsourced to another python module in the future

def discount_cash_flows(cash_flow_matrix, discount_rate, time_step_zero=0):
    num_paths, num_time_steps = cash_flow_matrix.shape

    # Calculate discount factors for each time step
    discount_factors = np.array([1 / (1 + discount_rate) ** (t - time_step_zero) for t in range(num_time_steps)])

    # Discount cash flows for each price path
    discounted_cash_flows = cash_flow_matrix * discount_factors

    # Sum the discounted cash flows along the time dimension to get the value at time zero
    cash_flow_at_time_zero = np.sum(discounted_cash_flows, axis=1)

    return cash_flow_at_time_zero

def create_cashflow_indicator_matrix(cash_flow_matrix):
    # Create a new matrix with the same shape as the cash flow matrix, filled with zeros
    indicator_matrix = np.zeros_like(cash_flow_matrix)

    # Set the elements to 1 where the cash flow matrix has non-zero values
    indicator_matrix[cash_flow_matrix != 0] = 1

    return indicator_matrix


def plot_invest_probability(cashflow_indicator_matrix):
    # Calculate the number of cash flows realized at each time step
    cashflows_per_time_step = np.sum(cashflow_indicator_matrix, axis=0)
    probability_of_invest = cashflows_per_time_step/cashflow_indicator_matrix.shape[0]

    # Generate time step labels (assuming you have 5 time steps)
    time_step_labels = np.arange(1, probability_of_invest.shape[0])

    # Plot the bar chart
    plt.bar(time_step_labels, probability_of_invest[1:])
    plt.xlabel('Time Step')
    plt.ylabel('Probability of investment')
    plt.title('Relative Number of Investments in Each Time Step')

    # Set the x-axis ticks to show only integer values
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.show()



