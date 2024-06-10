# Creates MathTools class with member variable total_calls
small = 1e-10


class MathTools:
    total_calls = 0
    # Constructor

    def __init__(self):
        self.total_calls = 0

    def derivative(self, f, x):
        self.total_calls += 1
        # For now we will take h to be a small value maybe
        h = small
        return (f(x + h) - f(x)) / h

    def gradient(self, f, x):
        self.total_calls += 1
        ans = []
        for i in range(len(x)):
            y = x.copy()
            y[i] += small
            ans.append((f(y) - f(x)) / small)
        return ans
