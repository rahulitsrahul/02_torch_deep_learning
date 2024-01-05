import matplotlib.pyplot as plt
import numpy as np

# Define two probability distributions
# These are just examples, you can use your own distributions
P = np.array([0.3, 0.4, 0.3])  # Probability distribution P
Q = np.array([0.5, 0.2, 0.3])  # Probability distribution Q

# Ensure that the distributions sum up to 1
P = P / np.sum(P)
Q = Q / np.sum(Q)

# Calculate KL divergence manually
kl_divergence = np.sum(P * np.log(P / Q))
print(f"KL Divergence from P to Q: {kl_divergence:.4f}")

# Plotting the distributions
x = np.arange(len(P))  # X-axis for the distributions

plt.figure(figsize=(8, 5))

plt.bar(x, P, width=0.4, align='center', alpha=0.7, label='P')
plt.bar(x + 0.4, Q, width=0.4, align='center', alpha=0.7, label='Q')

plt.xlabel('Events')
plt.ylabel('Probabilities')
plt.title('Probability Distributions P and Q')
plt.xticks(x + 0.2, ['Event 1', 'Event 2', 'Event 3'])
plt.legend()
plt.tight_layout()

plt.show()
