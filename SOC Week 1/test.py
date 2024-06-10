from ass1 import MathTools


def testForDerivative(x):
    # Assume that x will always be a number
    # A simple example function, might as well be anything more complex
    return x**2 + 2 * x + 3


def testForGradient(W):
    # Assume that W will always be a list of numbers with at least 1 element
    # A simple example function, might as well be anything more complex
    N = len(W)
    value = 0
    for i in range(N // 2):
        value += W[i] * W[N - i - 1]
    return value


tool = MathTools()

# Example usage with outputs on my device, might vary slightly on yours
print(tool.derivative(testForDerivative, 3))  # 8.00000000005241
print(
    tool.gradient(testForGradient, [1, 2, 3])
)  # [2.9999999999974487, 0.0, 1.0000000000065512]
print(tool.total_calls)  # 2

